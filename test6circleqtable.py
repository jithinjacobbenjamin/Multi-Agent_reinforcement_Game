import numpy as np  # for array stuff and random
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.
import math

style.use("ggplot")  # setting our style!
SIZE = 50    #environment diameter size

HM_EPISODES = 210000 #The game intervals when a game is shown in the environment
MOVE_PENALTY = 1 # this is the running inside payoff f
FOOD_REWARD = 200 #this is the boundary payoff value F
epsilon = 0.5  # randomness
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 10000  # how often to play through env visually.
z = 0
e=2  #epsilon distance from the token’s position
halfe = e/2
start_q_table = None  # if we have a pickled Q table, we'll put the filename of it here.
start_q_table2 = None
LEARNING_RATE = 0.1 # This can be between 0 and 1, the higher the number the better the learning rate base on the size of environment
DISCOUNT = 0.95 #This factor is kept at 0.95 to have a finite option of randomness for the agent during initial runs

PLAYER_N = 1
CIRCLE_N = 2
val=int(SIZE/2)
plotval=int(val/2)


d = {1: (255, 175, 0), 2: (0, 175, 255)} # blueish color for the token t in the game

if (SIZE % 2) == 0:
    flt = SIZE / 2
    bb = int(flt-1)
else:
    flt = SIZE / 2
    bb = int(flt)

class Blob:
    def __init__(self):
        self.x = np.random.randint(plotval, bb+plotval)
        self.y = np.random.randint(plotval, bb+plotval)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        '''
        Gives us 16 total movement options. (0,1,2,3....15)
        moving within the region epsilon
        '''

        if choice == 0:
            self.move(x=0.707, y=0.707)
        elif choice == 1:
            self.move(x=-0.707, y=-0.707)
        elif choice == 2:
            self.move(x=-0.707, y=0.707)
        elif choice == 3:
            self.move(x=0.707, y=-0.707)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=0, y=-1)
        elif choice == 6:
            self.move(x=-1, y=0)
        elif choice == 7:
            self.move(x=0, y=1)
        elif choice == 8:
            self.move(x=2, y=0)
        elif choice == 9:
            self.move(x=1.414, y=-1.414)
        elif choice == 10:
            self.move(x=0, y=-2)
        elif choice == 11:
            self.move(x=-1.414, y=-1.414)
        elif choice == 12:
            self.move(x=-2, y=0)
        elif choice == 13:
            self.move(x=-1.414, y=1.414)
        elif choice == 14:
            self.move(x=0, y=2)
        elif choice == 15:
            self.move(x=1.414, y=1.414)


            # Makes random moves when there is no probability data in the
            # Q-table for the current position of the token

    def move(self, x=False, y=False):
        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-e, e+1)
        else:
            self.x += x*(halfe)
        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-e, e+1)
        else:
            self.y += y*(halfe)

        # If we are out of bounds, this means that agent II achieved his goal!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

        # Initializing the Q-Table for Agent I

if start_q_table is None:
    # initialize the q-table#

    q_table = {}

    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            q_table[((i, ii))] = [np.random.uniform(-17, 0) for i in range(16)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


        # Initializing the Q-Table for Agent II


if start_q_table2 is None:
    # initialize the q-table2#

    q_table2 = {}

    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            q_table2[((i, ii))] = [np.random.uniform(-17, 0) for i in range(16)]

else:
    with open(start_q_table2, "rb") as f:
        q_table2 = pickle.load(f)

episode_rewards = []


class Blobb:
    def __init__(self):
        self.x = bb
        self.y = bb

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blobb()
    if episode % SHOW_EVERY == 0:
        # print(f"on #{episode}, epsilon is {epsilon}")
        print(f"on #{episode}, mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        # print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")

        show = True
    else:
        show = False
    episode_reward = 0
    for i in range(50000):
        choices = np.random.binomial(1, 0.5)
        obs = (player - food)
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        env = cv2.circle(env, (25, 25), 26, (0, 0, 255), 1)     # Circular border to show the border of the environment


        if choices == 0:
            if np.random.random() > epsilon:
                # GET THE ACTION
                action = np.argmax(q_table[obs])
            else:
                action = np.random.randint(0, 16)
            # Take the action!
            player.action(action)

            a = int(math.pow ((bb - player.y), 2))
            b = int(math.pow ((player.x - bb), 2))
            z = math.sqrt(a+b)

            if z > bb:  # or player.x == :
                reward = FOOD_REWARD
            else:
                reward = -MOVE_PENALTY





            # if player.x == SIZE - 1 or player.y == SIZE - 1 or player.x == 0 or player.y == 0:  # or player.x == :
            #     reward = FOOD_REWARD
            # else:
            #     reward = -MOVE_PENALTY





            new_obs = (player - food)  # new observation
            max_future_q = np.max(q_table[new_obs])  # max Q value for this new obs    ## <---
            current_q = q_table[obs][action]  # current Q for our chosen action

            if reward == FOOD_REWARD:
                new_q = FOOD_REWARD
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[obs][action] = new_q
            if show:
                #env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
                # env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color



                if (player.x < SIZE and player.y < SIZE):
                    env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue

                img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
                img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
                cv2.imshow("image", np.array(img))  # show it!
                if reward == FOOD_REWARD:  # or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                    if cv2.waitKey(200) & 0xFF == ord('q'):
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            episode_reward += reward
            if reward == FOOD_REWARD:  # or reward == -ENEMY_PENALTY:
                break

        elif choices == 1:
            if np.random.random() > epsilon:
                # GET THE ACTION
                action = np.argmin(q_table2[obs])
            else:
                action = np.random.randint(0, 16)
            # Take the action!
            player.action(action)

            if player.x == food.x and player.y == food.y:
                reward = FOOD_REWARD
            else:
                reward = -MOVE_PENALTY

            new_obs = (player - food)  # new observation
            max_future_q = np.min(q_table2[new_obs])  # max Q value for this new obs    ## <---
            current_q = q_table2[obs][action]  # current Q for our chosen action

            if reward == FOOD_REWARD:
                new_q = FOOD_REWARD
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table2[obs][action] = new_q
            if show:
                env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size

                if (player.x < SIZE and player.y < SIZE):
                    env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue

                img = Image.fromarray(env,
                                      'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
                img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
                cv2.imshow("image", np.array(img))  # show it!
                if reward == FOOD_REWARD:  # or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            episode_reward += reward
            if reward == FOOD_REWARD:  # or reward == -ENEMY_PENALTY:
                break
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)