"""User-sim three-agent pipeline (doc/UserSim_多轮Query在线生成.md).

Three collaborating agents construct multi-turn training data online during
rollout, after each winner is synced:

  - Observer   (no persona): objective state report from the winner (§3.3 / §7.3)
  - Questioner (persona)    : next follow-up query from the report (§3.3 / §7.4)
  - Reward     (frozen judge): observation-grounded score (§6 / §7.4)

Plus the 16-persona library (§3.5 / §7.5) and the patience failure mechanism
(§3.6.5). The session driver is rollout/simulated_session.run_simulated_session.

All modules are verl/Ray-free and unit-testable with mock clients.
"""

from agents.observer import Observer, parse_observation_report
from agents.personas import PERSONAS, sample_persona
from agents.questioner import END_SESSION, PatienceTracker, Questioner
from agents.reward import score_followup
from agents.schema import ObservationReport, Persona

__all__ = [
    "Observer",
    "parse_observation_report",
    "Questioner",
    "PatienceTracker",
    "END_SESSION",
    "score_followup",
    "ObservationReport",
    "Persona",
    "PERSONAS",
    "sample_persona",
]
