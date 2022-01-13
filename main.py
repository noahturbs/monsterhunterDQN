import dqn
import hotkeys
import memory

import pickle
import random
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


#setting: this turns gpu off.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#call to where your pc's autohotkey.exe is, so python can call it properly.
ahk = AHK(executable_path="C:\Program Files\AutoHotkey\AutoHotkey.exe")

time.sleep(1)
procname = "dolphin"

state = np.array([None]*22)

#loads from pickle, which is the machine learning model
with open('model_pickle', 'rb') as f:
    dqn_agent = pickle.load(f)

#USE THIS LINE BELOW IF YOU DON't HAVE PICKLE FILE
#dqn_agent = dqn.DQN()

#list of values to keep track of: ex hp, player location, etc

new_state= np.reshape([None]*22, [1,22])
empty_state= np.reshape([None]*22, [1,22])

keylistdown = hotkeys.returnkeylistdown()
keylistup = hotkeys.returnkeylistup()

time.sleep(1)



while(True):

    #outer loop: setup for dolphin!

    ahk.click(357, 749)#open dolphin
    time.sleep(10)

    #alternative way to find dolphin:
    #win = ahk.find_window(title=b'Dolphin 5.0-10222')
    #win.move(x=865, y=0, width=500, height=500)

    #(1058, 153)#select game.
    ahk.click(1058, 153)
    time.sleep(1)

    #(1015, 94) play button
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

    # this is what pm is:        pm = pymem.Pymem(procname)
    memory.update_memory_list(state[0],a,pm)
    new_state=state
    tempstate = np.reshape(np.array([None]*22), [1,22])

    action = 0

    #inner loop. runs the game frame by frame. checks for death count, boss hp being 0
    while(not(new_state[0][14]<1 or new_state[0][6]==3)):

        tempaddr = pm.read_bytes(a + 0x806CC2B8, 8) # 8 bytes, timer decrementing. if <1 then queststate = done

        #processes weird hex value from a string of letters to number.
        timer, = struct.unpack('!d', tempaddr)
        if(timer<81000):
            break

        action = dqn.dqn_agent.act(state)

        threaddown = threading.Thread(target=ahk.run_script, args=(keylistdown[action][0],))
        threaddown.start()

        threadup = threading.Thread(target=ahk.run_script, args=(keylistup[action][0],))

        threaddown.join()
        threadup.start()
        new_state = np.reshape([None]*22, [1,22])
        memory.update_memory_list(new_state[0],a,pm)

        #for t in keylist[action]:
        #    ahk.key_down(t)

        #PRINTS OUT DEATH COUNT, BOSS HP, QUEST TIMER
        print(new_state[0][6], new_state[0][14],   timer)


        dqn.dqn_agent.remember(state, action, state[0][14]-new_state[0][14], new_state, False)  #almost 0ms

        dqn.dqn_agent.replay()                      #~70ms

        dqn.dqn_agent.target_train()                #1~3ms


        state = new_state
        threadup.join()

    #state = new_state
    ahk.click(1089, 92)#CLICK STOP
    time.sleep(5)
    ahk.click(1328, 21)#CLICK CLOSE
    time.sleep(140)
    #pm.close_process()
    time.sleep(10)

    #saves machine learning model in pickle
    with open('model_pickle', 'wb') as f:
        pickle.dump(dqn_agent,f)
