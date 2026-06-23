import queue
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import config_wizard
from orchestrator import Orchestrator


class FakeLed:
    def __init__(self, index, events):
        self.index = index
        self.events = events

    def flash(self, color, duration_ms, brightness=100):
        self.events.append(("flash", self.index, color, duration_ms))
        start = float(len(self.events))
        return start, start + duration_ms / 1000.0


class FakeLedController:
    def __init__(self, events):
        self.led1 = FakeLed(1, events)
        self.led2 = FakeLed(2, events)

    def get_led(self, index):
        return self.led1 if index == 1 else self.led2


def make_orchestrator(schedule):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.sched = schedule
    orchestrator.hex_queue_1 = queue.Queue()
    orchestrator.ts_queue_1 = queue.Queue()
    orchestrator.hex_queue_2 = queue.Queue()
    orchestrator.ts_queue_2 = queue.Queue()
    orchestrator.events = []
    orchestrator.leds = FakeLedController(orchestrator.events)
    return orchestrator


class TerminalWizardTests(unittest.TestCase):
    def test_dual_defaults_to_three_flashes_and_can_add_more(self):
        answers = iter([
            "#FF0000", "",       # flash 1, default duration
            "#00FF00", "400",    # flash 2
            "#0000FF", "500",    # flash 3
            "y",                 # add fourth flash
            "#FFFFFF", "600",
            "n",                 # stop adding
            "900",               # gap between flashes
        ])
        with patch("builtins.input", side_effect=lambda _prompt: next(answers)):
            schedule = config_wizard._collect_dual_schedule()

        self.assertEqual(len(schedule["flashes"]), 4)
        self.assertEqual(
            schedule["flashes"],
            [
                {"hex": "#FF0000", "duration_ms": 300},
                {"hex": "#00FF00", "duration_ms": 400},
                {"hex": "#0000FF", "duration_ms": 500},
                {"hex": "#FFFFFF", "duration_ms": 600},
            ],
        )
        self.assertEqual(schedule["gap_ms"], 900)

    def test_sequential_defaults_to_three_rounds(self):
        answers = iter([
            "",          # default 3 rounds
            "#FFFF00",
            "350",
            "450",
            "1000",
        ])
        with patch("builtins.input", side_effect=lambda _prompt: next(answers)):
            schedule = config_wizard._collect_sequential_schedule("left_to_right")

        self.assertEqual(schedule, {
            "rounds": 3,
            "hex": "#FFFF00",
            "duration_ms": 350,
            "inner_pause_ms": 450,
            "gap_ms": 1000,
        })


class OrchestratorControlModeTests(unittest.TestCase):
    def test_dual_flashes_both_leds_for_every_configured_flash(self):
        orchestrator = make_orchestrator({
            "flashes": [
                {"hex": "#FF0000", "duration_ms": 100},
                {"hex": "#0000FF", "duration_ms": 200},
                {"hex": "#FFFFFF", "duration_ms": 300},
            ],
            "gap_ms": 800,
        })

        with patch("orchestrator.time.sleep") as sleep:
            orchestrator._run_dual()

        led1 = [event[2:] for event in orchestrator.events if event[1] == 1]
        led2 = [event[2:] for event in orchestrator.events if event[1] == 2]
        expected = [
            ("#FF0000", 100),
            ("#0000FF", 200),
            ("#FFFFFF", 300),
        ]
        self.assertEqual(led1, expected)
        self.assertEqual(led2, expected)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(orchestrator.hex_queue_1.qsize(), 3)
        self.assertEqual(orchestrator.hex_queue_2.qsize(), 3)

    def test_left_to_right_flashes_left_then_right_each_round(self):
        orchestrator = make_orchestrator({
            "rounds": 3,
            "hex": "#00FF00",
            "duration_ms": 300,
            "inner_pause_ms": 500,
            "gap_ms": 800,
        })

        timeline = orchestrator.events
        with patch(
            "orchestrator.time.sleep",
            side_effect=lambda seconds: timeline.append(("sleep", seconds)),
        ):
            orchestrator._run_sequential(first_led=1, second_led=2)

        flashes = [event[1] for event in timeline if event[0] == "flash"]
        self.assertEqual(flashes, [1, 2, 1, 2, 1, 2])
        self.assertEqual(
            [event for event in timeline if event[0] == "sleep"],
            [
                ("sleep", 0.5), ("sleep", 0.8),
                ("sleep", 0.5), ("sleep", 0.8),
                ("sleep", 0.5),
            ],
        )

    def test_right_to_left_flashes_right_then_left_each_round(self):
        orchestrator = make_orchestrator({
            "rounds": 3,
            "hex": "#FF00FF",
            "duration_ms": 300,
            "inner_pause_ms": 250,
            "gap_ms": 700,
        })

        timeline = orchestrator.events
        with patch(
            "orchestrator.time.sleep",
            side_effect=lambda seconds: timeline.append(("sleep", seconds)),
        ):
            orchestrator._run_sequential(first_led=2, second_led=1)

        flashes = [event[1] for event in timeline if event[0] == "flash"]
        self.assertEqual(flashes, [2, 1, 2, 1, 2, 1])


if __name__ == "__main__":
    unittest.main()
