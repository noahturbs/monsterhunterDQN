def returnkeylistdown():
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
    return keylistdown
def returnkeylistup():
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
    return keylistup
