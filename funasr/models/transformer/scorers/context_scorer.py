from funasr.models.transformer.scorers.scorer_interface import BatchPartialScorerInterface
import torch

class ContextScorer(BatchPartialScorerInterface):
    """
    Partial scorer based on funasr.utils.context_graph.ContextGraph.
    仅靠前缀树给热词奖励 (不依赖 CTC)。
    """

    def __init__(self, context_graph):
        self.context_graph = context_graph          # ContextGraph 实例

    # —— BatchPartialScorerInterface 实现 ——
    def init_state(self, x):
        # 初始状态：根节点
        return self.context_graph.root

    def select_state(self, state, i, new_id=None):
        # beam 内部按 index 选子状态
        return state[i]

    def score_partial(self, y, ids, state, x):
        """
        y    : prefix (unused)
        ids  : Tensor[K]  备选 token id
        state: ContextState
        x    : encoder out (unused)
        """
        scores, next_states = [], []
        for tok in ids:
            bonus, nxt = self.context_graph.forward_one_step(state, tok.item())
            scores.append(bonus)
            next_states.append(nxt)
        scores = torch.tensor(scores, device=x.device, dtype=x.dtype)
        return scores, next_states

    def final_score(self, state):
        # 结束时补偿未闭合前缀的分数
        bonus, final_state = self.context_graph.finalize(state)
        return bonus, final_state 