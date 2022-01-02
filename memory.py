import pymem
import struct

def update_memory_list(addrlist, e, z):
    #temp is a list. return a list.
    #rsi = 0x9014AB40
    a =int(e)
    pm=z
    #pm is from pymem library
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
