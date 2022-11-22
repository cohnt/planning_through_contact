from typing import Dict
import numpy as np
import networkx as nx
from irs_rrt.rrt_params import IrsRrtParams
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from pydrake.all import AngleAxis, Quaternion, RotationMatrix
from qsim.simulator import (
    QuasistaticSimulator,
    QuasistaticSimParameters,
    GradientMode,
    ForwardDynamicsMode,
)


class ReachableSet:
    """
    Computation class that computes parameters and metrics of reachable sets.
    """

    def __init__(
        self,
        q_dynamics: QuasistaticDynamics,
        params: IrsRrtParams,
        q_dynamics_p: QuasistaticDynamicsParallel = None,
    ):
        self.q_dynamics = q_dynamics
        if q_dynamics_p is None:
            q_dynamics_p = QuasistaticDynamicsParallel(self.q_dynamics)
        self.q_dynamics_p = q_dynamics_p

        self.q_u_indices_into_x = self.q_dynamics.get_q_u_indices_into_x()

        self.params = params
        self.n_samples = self.params.n_samples
        self.std_u = self.params.std_u
        self.regularization = self.params.regularization

        # QuasistaticSimulationParams
        self.q_sim_params = QuasistaticSimulator.copy_sim_params(
            self.q_dynamics.q_sim_params_default
        )
        self.q_sim_params.gradient_mode = GradientMode.kBOnly

    def calc_exact_Bc(self, q, ubar):
        """
        Compute exact dynamics.
        """
        self.q_sim_params.forward_mode = ForwardDynamicsMode.kQpMp

        x = q[None, :]
        u = ubar[None, :]
        (
            x_next,
            A,
            B,
            is_valid,
        ) = self.q_dynamics_p.q_sim_batch.calc_dynamics_parallel(
            x, u, self.q_sim_params
        )

        c = np.array(x_next).squeeze(0)
        B = np.array(B).squeeze(0)
        return B, c

    def calc_bundled_Bc_randomized(self, q, ubar):
        self.q_sim_params.gradient_mode = GradientMode.kBOnly
        self.q_sim_params.forward_mode = ForwardDynamicsMode.kSocpMp

        x_batch = np.tile(q[None, :], (self.n_samples, 1))
        u_batch = np.random.normal(
            ubar, self.std_u, (self.params.n_samples, self.q_dynamics.dim_u)
        )

        (
            x_next_batch,
            A_batch,
            B_batch,
            is_valid_batch,
        ) = self.q_dynamics_p.q_sim_batch.calc_dynamics_parallel(
            x_batch, u_batch, self.q_sim_params
        )

        if np.sum(is_valid_batch) == 0:
            raise RuntimeError("Cannot compute B and c hat for reachable sets.")

        B_batch = np.array(B_batch)
        x_next_batch = np.array(x_next_batch)

        chat = np.mean(x_next_batch[is_valid_batch], axis=0)
        Bhat = np.mean(B_batch[is_valid_batch], axis=0)
        return Bhat, chat

    def calc_bundled_Bc_randomized_zero_numpy(self, q, ubar):
        self.q_sim_params.gradient_mode = GradientMode.kNone
        self.q_sim_params.forward_mode = ForwardDynamicsMode.kSocpMp

        x_batch = np.tile(q[None, :], (self.n_samples, 1))
        u_batch = np.random.normal(
            ubar, self.std_u, (self.params.n_samples, self.q_dynamics.dim_u)
        )

        (
            x_next_batch,
            A_batch,
            B_batch,
            is_valid_batch,
        ) = self.q_dynamics_p.q_sim_batch.calc_dynamics_parallel(
            x_batch, u_batch, self.q_sim_params
        )

        if np.sum(is_valid_batch) == 0:
            raise RuntimeError("Cannot compute B and c hat for reachable sets.")

        chat = np.mean(x_next_batch[is_valid_batch], axis=0)
        Bhat = np.linalg.lstsq(
            u_batch[is_valid_batch] - ubar,
            x_next_batch[is_valid_batch] - chat,
            rcond=None,
        )[0].transpose()

        return Bhat, chat

    def calc_bundled_Bc_randomized_zero(self, q, ubar):
        Bhat, chat = self.q_dynamics_p.q_sim_batch.calc_Bc_lstsq(
            q, ubar, self.q_sim_params, self.std_u, self.params.n_samples
        )
        print(Bhat)
        return Bhat, chat

    def calc_bundled_Bc_analytic(self, q, ubar):
        q_next = self.q_dynamics.dynamics(
            x=q,
            u=ubar,
            forward_mode=ForwardDynamicsMode.kLogIcecream,
            gradient_mode=GradientMode.kBOnly,
        )

        Bhat = self.q_dynamics.q_sim.get_Dq_nextDqa_cmd()
        return Bhat, q_next

    def calc_metric_parameters(self, Bhat, chat):
        cov = Bhat @ Bhat.T + self.params.regularization * np.eye(
            self.q_dynamics.dim_x
        )
        mu = chat
        return cov, mu

    def calc_unactuated_metric_parameters(self, Bhat, chat):
        """
        Bhat: (n_a + n_u, n_a)
        """
        Bhat_u = Bhat[self.q_u_indices_into_x, :]
        cov_u = Bhat_u @ Bhat_u.T + self.params.regularization * np.eye(
            self.q_dynamics.dim_x - self.q_dynamics.dim_u
        )

        return cov_u, chat[self.q_u_indices_into_x]

    def calc_bundled_dynamics(self, Bhat, chat, du):
        xhat_next = Bhat.dot(du) + chat
        return xhat_next

    def calc_bundled_dynamics_batch(self, Bhat, chat, du_batch):
        xhat_next_batch = Bhat.dot(du_batch.transpose()).transpose() + chat
        return xhat_next_batch

    def calc_node_metric(self, covinv, mu, q_query):
        return (q_query - mu).T @ covinv @ (q_query - mu)

    def calc_node_metric_batch(self, covinv, mu, q_query_batch):
        batch_error = q_query_batch - mu[None, :]
        intsum = np.einsum("Bj,ij->Bi", batch_error, covinv)
        metric_batch = np.einsum("Bi,Bi->B", intsum, batch_error)
        return metric_batch
