import pickle
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
from ahk.window import Window

import pymem
import struct
import numpy as np
import time
from ahk import AHK
import datetime
import os
import threading
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ahk = AHK(executable_path="C:\Program Files\AutoHotkey\AutoHotkey.exe")

#print(ahk.mouse_position)
#ahk.click(357, 749)
time.sleep(1)
#win = ahk.find_window(title=b'Dolphin 5.0-10222')

procname = "dolphin"
#pm = pymem.Pymem(procname)
#base=  pm.process_base.lpBaseOfDll
#a = 0
#a = pm.read_bytes(base+0xF63A90, 8)
#a =int.from_bytes(a, byteorder='little')
class DQN:
    def __init__(self):
        self.memory  = deque(maxlen=500)

        self.gamma = 0.993           #controls reward
        self.epsilon = 1.0          #controls randomness
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.01   #large is good at beginning/generalization. small is good to optimize but bad at finding a solution
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
            model   = Sequential()
            #state_shape  = 23 # however many memory locations we are watching.
            model.add(Dense(22, input_shape=(22,) , activation="relu"))
            #model.add(Dense(46, activation="relu"))
            model.add(Dense(44, activation="relu"))
            model.add(Dense(44, activation="relu"))

            model.add(Dense(38))# however many possible moves we discretized. 54. final
            model.compile(loss="mean_squared_error",
                optimizer=Adam(lr=self.learning_rate))
            return model

    def act(self, state):
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            #print(np.argmax(self.model.predict(state)[0]))
            if np.random.random() < self.epsilon:
                return np.random.randint(0,37)                            #return self.env.action_space.sample()
            return np.argmax(self.model.predict(state)[0])


    def remember(self, state, action, reward, new_state, done):
            self.memory.append([state, action, reward, new_state, done])

    def replay(self):
            batch_size = 32
            if len(self.memory) < batch_size:
                return

            samples = random.sample(self.memory, batch_size)
            for sample in samples:
                state, action, reward, new_state, done = sample
                target = self.target_model.predict(state)
                if done:
                    target[0][action] = reward
                else:
                    Q_future = max(self.target_model.predict(new_state)[0])
                    target[0][action] = reward + Q_future * self.gamma
                self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)
    def set_epsilon(self, value):
        self.epsilon = value


def update_memory_list(addrlist, e, z):
    #temp is a list. return a list.
    #rsi = 0x9014AB40
    a =int(e)
    pm=z

    '''
    f = pm.read_bytes(a + 0x806CC2B8, 8)
    d2, = struct.unpack('!d', f)
    #int.from_bytes(f, byteorder='little')
    print(d2)
    '''
    tempaddr = pm.read_bytes(a + 0x901481F6, 1) # 1 byte, player map pos
    addrlist[0] = int.from_bytes(tempaddr, byteorder='little')

    tempaddr = pm.read_bytes(a + 0x9014AB7C, 4) #4 bytes, float. player x position
    #addrvalue, = struct.unpack('!f', tempaddr)
    addrlist[1], = struct.unpack('!f', tempaddr)

    tempaddr = pm.read_bytes(a + 0x9014AB84, 4) #4 bytes, float. player x position
    #addrvalue, = struct.unpack('!f', tempaddr)
    addrlist[2], = struct.unpack('!f', tempaddr)

    tempaddr = pm.read_bytes(a + 0x9014AB9A, 2) # 2 bytes, short. player orientation.
    #addrvalue, = struct.unpack('!H', tempaddr)
    addrlist[3], = struct.unpack('!H', tempaddr)

    tempaddr = pm.read_bytes(a + 0x900AD768, 4) #4 bytes, float. player camera x
    #addrvalue, = struct.unpack('!f', tempaddr)
    addrlist[4], = struct.unpack('!f', tempaddr)

    tempaddr = pm.read_bytes(a + 0x900AD770, 4) #4 bytes, float. player camera x
    #addrvalue, = struct.unpack('!f', tempaddr)
    addrlist[5], = struct.unpack('!f', tempaddr)

    tempaddr = pm.read_bytes(a + 0x9014AF86, 1) # 1 byte, death count. if ==3 then queststate = done
    #addrvalue = int.from_bytes(tempaddr, byteorder='little')
    addrlist[6] = int.from_bytes(tempaddr, byteorder='little')


    tempaddr = pm.read_bytes(a + 0x9014B0AD, 1) # 1 byte, weapon sharpness
    #addrvalue = int.from_bytes(tempaddr, byteorder='little')
    addrlist[7] = int.from_bytes(tempaddr, byteorder='little')

    tempaddr = pm.read_bytes(a + 0x9014AEB1, 1) # 1 byte, green hp
    #addrvalue = int.from_bytes(tempaddr, byteorder='little')
    addrlist[8] = int.from_bytes(tempaddr, byteorder='little')

    tempaddr = pm.read_bytes(a + 0x9014AEB7, 1) # 1 byte, red hp
    #addrvalue = int.from_bytes(tempaddr, byteorder='little')
    addrlist[9] = int.from_bytes(tempaddr, byteorder='little')

    tempaddr = pm.read_bytes(a + 0x9014AEB8, 2) # 2 bytes, player stamina
    #addrvalue, = struct.unpack('!H', tempaddr)
    addrlist[10], = struct.unpack('!H', tempaddr)

    tempaddr = pm.read_bytes(a + 0x9014AC62, 1) # 1 byte, hammer charge time
    #addrvalue = int.from_bytes(tempaddr, byteorder='little')
    addrlist[11] = int.from_bytes(tempaddr, byteorder='little')


    #NOW DO MONSTER'S INFORMATION STUFF

    tempaddr = pm.read_bytes(a + 0x9014E008, 2) # 2 bytes, short. monster x pos
    #addrvalue, = struct.unpack('!H', tempaddr)
    addrlist[12], = struct.unpack('!H', tempaddr)

    tempaddr = pm.read_bytes(a + 0x9014E010, 2) # 2 bytes, short. monster y pos
    #addrvalue, = struct.unpack('!H', tempaddr)
    addrlist[13], = struct.unpack('!H', tempaddr)

    tempaddr = pm.read_bytes(a + 0x9014E622, 2) # 2 bytes, short. monster HP if <1 then queststate= done
    #addrvalue, = struct.unpack('!H', tempaddr)
    addrlist[14], = struct.unpack('!H', tempaddr)

    tempaddr = pm.read_bytes(a + 0x806AC8B5, 1) # 1 byte, monster rage
    #addrvalue = int.from_bytes(tempaddr, byteorder='little')
    addrlist[15] = int.from_bytes(tempaddr, byteorder='little')

    tempaddr = pm.read_bytes(a + 0x9014E042, 2) # 2 bytes, short. monster orientation
    #addrvalue, = struct.unpack('!H', tempaddr)
    addrlist[16], = struct.unpack('!H', tempaddr)

    tempaddr = pm.read_bytes(a + 0x9014E7BF, 1) # 1 byte, monster ko meter
    #addrvalue = int.from_bytes(tempaddr, byteorder='little')
    addrlist[17] = int.from_bytes(tempaddr, byteorder='little')

    tempaddr = pm.read_bytes(a + 0x9014E6BA, 2) # 2 bytes, short. monster head stagger (starts at 180. decrements.)
    #addrvalue, = struct.unpack('!H', tempaddr)
    addrlist[18], = struct.unpack('!H', tempaddr)

    tempaddr = pm.read_bytes(a + 0x9014E6B4, 1) # 1 bytes, short. monster head stagger count
    #addrvalue = int.from_bytes(tempaddr, byteorder='little')
    addrlist[19] = int.from_bytes(tempaddr, byteorder='little')

    tempaddr = pm.read_bytes(a + 0x9014AE5A, 1) # 1 bytes, short. some kind of player animation state stuff
    #addrvalue = int.from_bytes(tempaddr, byteorder='little')
    addrlist[20] = int.from_bytes(tempaddr, byteorder='little')

    tempaddr = pm.read_bytes(a + 0x9014AE5B, 1) # 1 bytes, short. some kind of player animation state stuff, a second one
    #addrvalue = int.from_bytes(tempaddr, byteorder='little')
    addrlist[21] = int.from_bytes(tempaddr, byteorder='little')









keylistdown = []
keylistdown.append(['SendInput {. down}'])
keylistdown.append(['SendInput {a DOWN}{. down}'])
keylistdown.append(['SendInput {a DOWN}{w DOWN}{. down}'])
keylistdown.append(['SendInput {w DOWN}{. down}'])
keylistdown.append(['SendInput {w DOWN}{d DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{s DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{a DOWN}{. down}'])

keylistdown.append(['SendInput {u DOWN}{. down}']) #rolls, 'u'
keylistdown.append(['SendInput {a DOWN}{u DOWN}{. down}'])
keylistdown.append(['SendInput {a DOWN}{w DOWN}{u DOWN}{. down}'])
keylistdown.append(['SendInput {w DOWN}{u DOWN}{. down}'])
keylistdown.append(['SendInput {w DOWN}{d DOWN}{u DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{u DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{s DOWN}{u DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{u DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{a DOWN}{u DOWN}{. down}'])


keylistdown.append(['SendInput {p DOWN}{. down}']) #running/charging, 'p' char
keylistdown.append(['SendInput {a DOWN}{p DOWN}{. down}'])
keylistdown.append(['SendInput {a DOWN}{w DOWN}{p DOWN}{. down}'])
keylistdown.append(['SendInput {w DOWN}{p DOWN}{. down}'])
keylistdown.append(['SendInput {w DOWN}{d DOWN}{p DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{p DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{s DOWN}{p DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{p DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{a DOWN}{p DOWN}{. down}'])

keylistdown.append(['SendInput {j DOWN}{. down}']) #hits, 'k' char
keylistdown.append(['SendInput {a DOWN}{j DOWN}{. down}'])
keylistdown.append(['SendInput {a DOWN}{w DOWN}{j DOWN}{. down}'])
keylistdown.append(['SendInput {w DOWN}{j DOWN}{. down}'])
keylistdown.append(['SendInput {w DOWN}{d DOWN}{j DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{j DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{s DOWN}{j DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{j DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{a DOWN}{j DOWN}{. down}'])


keylistdown.append(['SendInput {i DOWN}{. down}']) #sheathes, 'i' char, reduced amount.
'''
keylistdown.append(['SendInput {w DOWN}{i DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{i DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{i DOWN}{. down}'])
'''

keylistdown.append(['SendInput {p DOWN}{j DOWN}{k DOWN}{. down}']) #r x y ,'p' 'j' 'k' char
'''
keylistdown.append(['SendInput {a DOWN}{p DOWN}{j DOWN}{k DOWN}{. down}'])
keylistdown.append(['SendInput {a DOWN}{w DOWN}{p DOWN}{j DOWN}{k DOWN}{. down}'])
keylistdown.append(['SendInput {w DOWN}{p DOWN}{j DOWN}{k DOWN}{. down}'])
keylistdown.append(['SendInput {w DOWN}{d DOWN}{p DOWN}{j DOWN}{k DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{p DOWN}{j DOWN}{k DOWN}{. down}'])
keylistdown.append(['SendInput {d DOWN}{s DOWN}{p DOWN}{j DOWN}{k DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{p DOWN}{j DOWN}{k DOWN}{. down}'])
keylistdown.append(['SendInput {s DOWN}{a DOWN}{p DOWN}{j DOWN}{k DOWN}{. down}'])
'''


#####KEYLISTup

keylistup = []
keylistup.append(['SendInput{. up}'])
keylistup.append(['SendInput {a UP}{. up}'])
keylistup.append(['SendInput {a UP}{w UP}{. up}'])
keylistup.append(['SendInput {w UP}{. up}'])
keylistup.append(['SendInput {w UP}{d UP}{. up}'])
keylistup.append(['SendInput {d UP}{. up}'])
keylistup.append(['SendInput {d UP}{s UP}{. up}'])
keylistup.append(['SendInput {s UP}{. up}'])
keylistup.append(['SendInput {s UP}{a UP}{. up}'])

keylistup.append(['SendInput {u UP}{. up}']) #rolls, 'u'
keylistup.append(['SendInput {a UP}{u UP}{. up}'])
keylistup.append(['SendInput {a UP}{w UP}{u UP}{. up}'])
keylistup.append(['SendInput {w UP}{u UP}{. up}'])
keylistup.append(['SendInput {w UP}{d UP}{u UP}{. up}'])
keylistup.append(['SendInput {d UP}{u UP}{. up}'])
keylistup.append(['SendInput {d UP}{s UP}{u UP}{. up}'])
keylistup.append(['SendInput {s UP}{u UP}{. up}'])
keylistup.append(['SendInput {s UP}{a UP}{u UP}{. up}'])


keylistup.append(['SendInput {p UP}{. up}']) #running/charging, 'p' char
keylistup.append(['SendInput {a UP}{p UP}{. up}'])
keylistup.append(['SendInput {a UP}{w UP}{p UP}{. up}'])
keylistup.append(['SendInput {w UP}{p UP}{. up}'])
keylistup.append(['SendInput {w UP}{d UP}{p UP}{. up}'])
keylistup.append(['SendInput {d UP}{p UP}{. up}'])
keylistup.append(['SendInput {d UP}{s UP}{p UP}{. up}'])
keylistup.append(['SendInput {s UP}{p UP}{. up}'])
keylistup.append(['SendInput {s UP}{a UP}{p UP}{. up}'])

keylistup.append(['SendInput {j UP}{. up}']) #hits, 'k' char
keylistup.append(['SendInput {a UP}{j UP}{. up}'])
keylistup.append(['SendInput {a UP}{w UP}{j UP}{. up}'])
keylistup.append(['SendInput {w UP}{j UP}{. up}'])
keylistup.append(['SendInput {w UP}{d UP}{j UP}{. up}'])
keylistup.append(['SendInput {d UP}{j UP}{. up}'])
keylistup.append(['SendInput {d UP}{s UP}{j UP}{. up}'])
keylistup.append(['SendInput {s UP}{j UP}{. up}'])
keylistup.append(['SendInput {s UP}{a UP}{j UP}{. up}'])


keylistup.append(['SendInput {i UP}{. up}']) #sheathes, 'i' char, reduced amount.
keylistup.append(['SendInput {p UP}{j UP}{k UP}{. up}']) #r x y ,'p' 'j' 'k' char

state = np.array([None]*22)

#dqn_agent = DQN()
with open('model_pickle', 'rb') as f:
    dqn_agent = pickle.load(f)
new_state= np.reshape([None]*22, [1,22])
empty_state= np.reshape([None]*22, [1,22])
#win.close()
time.sleep(1)
#print(ahk.mouse_position)
#(1058, 153)#select game.
#(977, 94) play button
#time.sleep(1000)
while(True):
    #ahk.key_press('.') #advance frame
    ahk.click(357, 749)#open dolphin
    time.sleep(3)

    #win = ahk.find_window(title=b'Dolphin 5.0-10222')

    time.sleep(5)
    #win.move(x=865, y=0, width=500, height=500)
    time.sleep(1)
    #(1058, 153)#select game.
    #(1015, 94) play button
    ahk.click(1058, 153)
    time.sleep(1)

    ahk.click(1015, 94)
    time.sleep(15)
    ahk.key_press('.')
    time.sleep(2)
    #click on play



    ahk.key_press('.')

    time.sleep(10)
    ahk.key_press('F1')
    time.sleep(10)

    pm = pymem.Pymem(procname)

    time.sleep(4)
    base=  pm.process_base.lpBaseOfDll
    time.sleep(4)


    state = empty_state
    time.sleep(2)
    a = pm.read_bytes(base+0xF63A90, 8)
    a =int.from_bytes(a, byteorder='little')

    time.sleep(5)
    state= np.reshape(state, [1,22])
    ahk.key_press('.')
    time.sleep(4)

    update_memory_list(state[0],a,pm)
    new_state=state
    tempstate = np.reshape(np.array([None]*22), [1,22])

    action = 0
    while(not(new_state[0][14]<1 or new_state[0][6]==3)):

        tempaddr = pm.read_bytes(a + 0x806CC2B8, 8) # 8 bytes, timer decrementing. if <1 then queststate = done
        #addrvalue, = struct.unpack('!d', tempaddr)
        timer, = struct.unpack('!d', tempaddr)
        if(timer<81000):
            break

        #update_memory_list(tempstate[0])
        '''if(tempstate[0][7] == state[0][7]):
            time.sleep(0.001)
            ahk.run_script('SendInput .')dpsaawpjkdjaspawpjk
            #ahk.run_script('Sleep, 5')

            continue
'''
        #print('hi')
        #ahk.click()
        action = dqn_agent.act(state)


        #ahk.send_raw(keylist[action])
        #new_state=   empty_state
        #update_memory_list(new_state)
        #print(new_state)
        #new_state= np.reshape(new_state, [1,23])u.apjk.saj.dj.dp.saj.d.dj.aw

        #dd = datetime.datetime.now()


        threaddown = threading.Thread(target=ahk.run_script, args=(keylistdown[action][0],))
        threaddown.start()

        #bb = datetime.datetime.now()
        #delta = bb-dd
        #print((delta.total_seconds() * 1000))

        #print(keylistdown[action][0])
        threadup = threading.Thread(target=ahk.run_script, args=(keylistup[action][0],))

        #ahk.run_script(keylistup[action][0], blocking=True)
        #threading.Thread(target=ahk.key_down, args= (keylist[action][t],)) )
        #thread[t].start()
        #thread.start()

        #dd = datetime.datetime.now()


        #ahk.key_press('.') #advance frame
        #ahk_script = 'SendInput '

        #ahk.run_script(ahk_script, blocking=False)


        #dd = datetime.datetime.now()

        #bb = datetime.datetime.now()
        #delta = bb-dd
        #print((delta.total_seconds() * 1000))

        #somehow, key_press is taking a little too long.
        #thread = []
        #for t in keylist[action]:
        #    ahk.key_up(t)

        #for t in range(len(keylist[action])):
        #    thread.append(threading.Thread(target=ahk.key_up, args=(keylist[action][t],)) )
        #    thread[t].start()

        #dd = datetime.datetime.now()

        threaddown.join()

        #bb = datetime.datetime.now()
        #delta = bb-dd
        #print((delta.total_seconds() * 1000))

        threadup.start()
        #new_state = np.reshape(np.array([None]*22), [1,22])
        new_state = np.reshape([None]*22, [1,22])
        update_memory_list(new_state[0],a,pm)
        #state = new_state.ss


        #for t in keylist[action]:
        #    ahk.key_down(t)


        print(new_state[0][6], new_state[0][14],   timer)

        #reward = reward if not done else -20
        #calculate rewardpjkds
        #print(new_state[0])
        #stuff= new_state[0]
        #oldstuff = state[0]
        #print(stuff[7])
        #if we
        #reward = new_state[0][15]-state[0][15]

        #caculate whether we are done.

        #new_state = new_state.reshape(1,2)

        #dd = datetime.datetime.now()
        #print(state[0][14] - new_state[0][14])



        dqn_agent.remember(state, action, state[0][14]-new_state[0][14], new_state, False)  #almost 0ms

        #dd = datetime.datetime.now()

        #bb = datetime.datetime.now()
        #delta = bb-dd
        #print((delta.total_seconds() * 1000))




        dqn_agent.replay()                      #~70ms


        #dd = datetime.datetime.now()

        dqn_agent.target_train()                #1~3ms
        #bb = datetime.datetime.now()
        #delta = bb-dd
        #print((delta.total_seconds() * 1000))



        state = new_state
        threadup.join()
        #ahk.run_script(keylistdown[action][0],blocking = True)


        #for t in range(len(keylist[action])):
        #    thread[t].join()
    #state = new_state
    ahk.click(1089, 92)#CLICK STOP
    time.sleep(5)
    ahk.click(1328, 21)#CLICK CLOSE
    time.sleep(140)
    #pm.close_process()
    time.sleep(10)
    #for ddd in range(50):
    #    ahk.key_press('.')
    with open('model_pickle', 'wb') as f:
        pickle.dump(dqn_agent,f)
