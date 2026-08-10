from tcp_base import TcpEventBased
import numpy as np
import state_filter
import torch
# Note: do NOT create a module-level DQNAgent here. The training/testing
# code must instantiate the agent and pass it to `get_action` to avoid
# duplicate/unsynchronized agent instances.
filter = state_filter.Filter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RLTCP(TcpEventBased):
    """ACK-level TCP agent that uses multiplicative congestion-window actions."""

    def __init__(
        self,
        action_mode='default',
        action0_scale=5000.0,
        action1_add=340,
        action2_add=1500,
        action3_scale=1.2,
        action4_scale=0.6,
        action5_scale=0.8,
    ):
        super(RLTCP, self).__init__()
        self.action_mode = action_mode
        self.action0_scale = action0_scale
        self.action1_add = action1_add
        self.action2_add = action2_add
        self.action3_scale = action3_scale
        self.action4_scale = action4_scale
        self.action5_scale = action5_scale

    def get_action(self, state, obs, dqn_agent):
        """Return [new_ss_thresh, new_cwnd, action_index].

        Explicitly requires `dqn_agent` (model.DQNAgent instance).
        """
        if dqn_agent is None:
            raise ValueError("dqn_agent must be provided to get_action")

        mss = 340
        action_index = dqn_agent.selectAction(state)
        c_wnd = float(obs[5])
        ssThresh = float(obs[4])
        # print(action_index)

        # ACTION SPACE RE-DESIGNED FOR FAST RECOVERY
        # 0: custom multiplier or default double CWND
        # 1: custom add or default MSS
        # 2: custom add or default 1500 bytes
        # 3: custom multiplier or default 1.2x
        # 4: custom multiplier or default 0.6x
        # 5: custom multiplier or default 0.8x
        if action_index == 0:
            new_cwnd = c_wnd * self.action0_scale
        elif action_index == 1:
            new_cwnd = c_wnd + self.action1_add
        elif action_index == 2:
            new_cwnd = c_wnd + self.action2_add
        elif action_index == 3:
            new_cwnd = c_wnd * self.action3_scale
        elif action_index == 4:
            new_cwnd = c_wnd * self.action4_scale
        elif action_index == 5:
            new_cwnd = c_wnd * self.action5_scale

        new_cwnd = max(mss, int(np.ceil(new_cwnd)))
        # Use real division for half the cwnd, not integer floor division
        new_ss_thresh = max(mss, int(np.ceil(new_cwnd / 2.0)))

        return [new_ss_thresh, new_cwnd, action_index]