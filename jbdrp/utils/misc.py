import numpy as np
from copy import copy

def guess_star_fiber(image,usershift=0,fiber1=None,fiber2=None,fiber3=None):
    if fiber1 is None:
        _fiber1 = np.array([[70,150],[260,330],[460,520],[680,720],[900,930],[1120,1170],[1350,1420],[1600,1690],[1870,1980]])
    else:
        _fiber1 = copy(fiber1)
    if fiber2 is None:
        _fiber2 = np.array([[50,133],[240,320],[440,510],[650,710],[880,910],[1100,1150],[1330,1400],[1580,1670],[1850,1960]])
    else:
        _fiber2 = copy(fiber2)
    if fiber3 is None:
        _fiber3 = np.array([[30,120],[220,300],[420,490],[640,690],[865,890],[1090,1130],[1320,1380],[1570,1650],[1840,1940]])
    else:
        _fiber3 = copy(fiber3)
    _fiber1 += usershift
    _fiber2 += usershift
    _fiber3 += usershift

    fiber1_template = np.zeros(2048)
    for x1,x2 in _fiber1:
        fiber1_template[x1+10:x2-10] = 1
    fiber2_template = np.zeros(2048)
    for x1,x2 in _fiber2:
        fiber2_template[x1+10:x2-10] = 1
    fiber3_template = np.zeros(2048)
    for x1,x2 in _fiber3:
        fiber3_template[x1+10:x2-10] = 1
    flattened = np.nanmean(image,axis=1)

    # import matplotlib.pyplot as plt
    # plt.plot(flattened/np.nanmax(flattened),label="flat")
    # plt.plot(fiber1_template,label="0")
    # plt.plot(fiber2_template,label="1")
    # plt.plot(fiber3_template,label="2")
    # plt.show()

    return np.argmax([np.nansum(fiber1_template * flattened),
                       np.nansum(fiber2_template*flattened),
                       np.nansum(fiber3_template*flattened)])