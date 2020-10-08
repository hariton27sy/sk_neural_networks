import time
import gym
import numpy as np
# 0 - юг
# 1 - север
# 2 - восток
# 3 - запад
# 4 - подобрать
# 5 - высадить


class Agent:
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.policy = np.ones((state_n, action_n)) / action_n

    def get_action(self, state):
        prob = self.policy[state]
        action = np.random.choice(np.arange(self.action_n), p=prob)
        return int(action)

    def update_policy(self, elite_sessions):
        new_policy = np.zeros((self.state_n, self.action_n))

        for session in elite_sessions:
            for state, action in zip(session['states'], session['actions']):
                new_policy[state][action] += 1

        for state in range(self.state_n):
            if sum(new_policy[state]) == 0:
                new_policy[state] += 1 / self.action_n
            else:
                new_policy[state] /= sum(new_policy[state])

        self.policy = new_policy


def get_session(env, agent, session_len, visual=False):
    """Запуск одной игры"""
    session = {}
    states, actions = [], []
    total_reward = 0

    state = env.reset()
    for _ in range(session_len):
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)

        if visual:
            env.render()
            time.sleep(1)

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    session['states'] = states
    session['actions'] = actions
    session['total_reward'] = total_reward
    return session


def get_elite_sessions(sessions, q_param):
    total_rewards = np.array([session['total_reward'] for session in sessions])
    quantile = np.quantile(total_rewards, q_param)

    elite_sessions = []
    for session in sessions:
        if session['total_reward'] > quantile:
            elite_sessions.append(session)
    return elite_sessions


def run_episode(episode, env, agent, session_len, n_sesions, q_param):
    """Запуск одного эпизода и обновление policy агента"""
    sessions = [get_session(env, agent, session_len) for _ in range(n_sesions)]
    mean_total_reward = np.mean([session['total_reward'] for session in sessions])
    print(f"{episode}. Mean total reward = {mean_total_reward}")

    elite_sessions = get_elite_sessions(sessions, q_param)
    if len(elite_sessions) > 0:
        agent.update_policy(elite_sessions)


def main():
    env = gym.make("Taxi-v3")
    n_states = 500  # 25 позиций * 5 состояний пассажира (в одном из пунктов + в такси) * 4 точки назначения
    n_actions = 6  # Направления + посадить/высадить

    n_episodes = 50
    n_sessions = 500  # В каждом эпизоде необходимо побольше сессий, чтобы было хоть что-то для каждого начального состояния
    session_len = 120  # Много ходов для одной сессии не нужно, потому что сессии с большим количеством ходов сильно уменьшают счет
    q_param = 0.55  # Достаточно много различных состояний, поэтому в условиях неогромного количества сессий и эпизодов понижается квантиль
    print(f"n_episodes: {n_episodes}\nn_sessions: {n_sessions}\nsession_len: {session_len}\nq_param: {q_param}")
    agent = Agent(n_states, n_actions)

    for episode in range(n_episodes):
        run_episode(episode, env, agent, session_len, n_sessions, q_param)

    get_session(env, agent, session_len, visual=True)
    env.close()


if __name__ == "__main__":
    main()
