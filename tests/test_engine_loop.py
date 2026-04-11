from __future__ import annotations

from vrl.engine import ContinuousBatchPlanner, EngineLoop, Scheduler
from vrl.engine.types import RequestOutput, SchedulerRequest, SchedulerStatus


class _FeedbackIterationController:
    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        request.data = output.data

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        return True

    def apply_feedback(self, request: SchedulerRequest, data) -> None:
        request.metadata["feedback"] = data


class _FeedbackMailbox:
    def __init__(self, items: dict[str, object]) -> None:
        self._items = dict(items)

    def has(self, request_id: str) -> bool:
        return request_id in self._items

    def pop(self, request_id: str):
        return self._items.pop(request_id, None)


def test_feedback_mailbox_transitions_request():
    scheduler = Scheduler(
        batch_planner=ContinuousBatchPlanner(max_batch_size=1),
        iteration_controller=_FeedbackIterationController(),
    )
    request = SchedulerRequest(
        request_id="req-1",
        data={"step": 1},
        status=SchedulerStatus.WAITING_FEEDBACK,
    )
    scheduler.requests[request.request_id] = request
    mailbox = _FeedbackMailbox({"req-1": {"approved": True}})
    engine = EngineLoop(scheduler=scheduler, model_runner=object(), feedback_mailbox=mailbox)

    engine._check_feedback()

    assert request.status is SchedulerStatus.RUNNING
    assert request.metadata["feedback"] == {"approved": True}
    assert not mailbox.has("req-1")
