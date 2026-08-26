"""Deterministic grader for T058zh_painting_identification."""

from claw_eval_vendor.graders.base import AbstractGrader
from claw_eval_vendor.graders.image_qa_oracle import ImageQAOracleMixin


class PaintingIdentificationGrader(ImageQAOracleMixin, AbstractGrader):
    """Oracle-based image QA grader for T38_painting_identification."""
