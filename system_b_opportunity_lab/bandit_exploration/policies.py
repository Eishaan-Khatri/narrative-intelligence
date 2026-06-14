"""Simple exploration policies for item opportunity simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BanditPolicy:
    arms: list[str]
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def __post_init__(self) -> None:
        self.counts = {arm: 0 for arm in self.arms}
        self.rewards = {arm: 0.0 for arm in self.arms}

    def select_arm(self) -> str:
        raise NotImplementedError

    def update(self, arm: str, reward: float) -> None:
        self.counts[arm] += 1
        self.rewards[arm] += float(reward)

    def mean_reward(self, arm: str) -> float:
        n = self.counts[arm]
        return self.rewards[arm] / n if n else 0.0


@dataclass
class PopularityPolicy(BanditPolicy):
    popularity: dict[str, float] | None = None

    def select_arm(self) -> str:
        if not self.popularity:
            return self.arms[0]
        return max(self.arms, key=lambda arm: self.popularity.get(arm, 0.0))


@dataclass
class EpsilonGreedyPolicy(BanditPolicy):
    epsilon: float = 0.1

    def select_arm(self) -> str:
        if self.rng.random() < self.epsilon:
            return str(self.rng.choice(self.arms))
        return max(self.arms, key=lambda arm: self.mean_reward(arm))


@dataclass
class UCB1Policy(BanditPolicy):
    c: float = 1.5

    def select_arm(self) -> str:
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm
        total = sum(self.counts.values())
        return max(
            self.arms,
            key=lambda arm: self.mean_reward(arm) + self.c * np.sqrt(np.log(total) / self.counts[arm]),
        )


@dataclass
class ThompsonSamplingPolicy(BanditPolicy):
    alpha0: float = 1.0
    beta0: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.alpha = {arm: self.alpha0 for arm in self.arms}
        self.beta = {arm: self.beta0 for arm in self.arms}

    def select_arm(self) -> str:
        samples = {arm: self.rng.beta(self.alpha[arm], self.beta[arm]) for arm in self.arms}
        return max(self.arms, key=lambda arm: samples[arm])

    def update(self, arm: str, reward: float) -> None:
        bounded = float(np.clip(reward, 0.0, 1.0))
        super().update(arm, bounded)
        self.alpha[arm] += bounded
        self.beta[arm] += 1.0 - bounded
