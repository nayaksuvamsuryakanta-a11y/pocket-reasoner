"""The advantage is how far an answer's reward is above its group's average; GRPO pushes up answers with a positive advantage and pushes down answers with a negative one; libraries may also divide by the group's spread, but a zero stays zero."""


def advantages(rewards):
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


if __name__ == "__main__":
    groups = {
        "A": [0, 0, 0, 0, 0, 0, 0, 0],
        "B": [2.5, 0, 0, 0, 0, 0, 0, 0],
        "C": [3, 3, 3, 3, 3, 3, 3, 3],
        "D": [3, 3, 1, 1, 0.5, 0.5, 0, 0],
    }
    for name, group_rewards in groups.items():
        mean = sum(group_rewards) / len(group_rewards)
        group_advantages = advantages(group_rewards)
        formatted_advantages = ", ".join(format(value, ".2f") for value in group_advantages)
        print(f"{name}: rewards={group_rewards}, mean={mean}, advantages=[{formatted_advantages}]")