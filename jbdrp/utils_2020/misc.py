import numpy as np
from copy import copy

def guess_star_fiber(image,usershift=0,fiber1=None,fiber2=None,fiber3=None,fiber4=None):
    if fiber1 is None:
        _fiber1 = np.array([[70, 150], [260, 330], [460, 520], [680 - 10, 720 + 10], [900 - 15, 930 + 15], [1120 - 5, 1170 + 5],
             [1350, 1420], [1600, 1690], [1870, 1980]]) + 15
    else:
        _fiber1 = copy(fiber1)
    if fiber2 is None:
        _fiber2 = np.array([[50, 133], [240, 320], [440, 510], [650, 710], [880 - 15, 910 + 15], [1100 - 5, 1150 + 5], [1330, 1400],
             [1580, 1670], [1850, 1960]]) + 15
    else:
        _fiber2 = copy(fiber2)
    if fiber3 is None:
        _fiber3 = np.array([[30, 120], [220, 300], [420, 490], [640 - 5, 690 + 5], [865 - 20, 890 + 20], [1090 - 10, 1130 + 10],
             [1320, 1380], [1570, 1650], [1840, 1940]]) + 10
    else:
        _fiber3 = copy(fiber3)
    if fiber4 is None:
        _fiber4 = np.array([[30, 120], [220, 300], [420, 490], [640 - 5, 690 + 5], [865 - 20, 890 + 20], [1090 - 10, 1130 + 10],
             [1320, 1380], [1570, 1650], [1840, 1940]]) - 10
    else:
        _fiber4 = copy(fiber4)
    _fiber1 += usershift
    _fiber2 += usershift
    _fiber3 += usershift
    _fiber4 += usershift

    fiber1_template = np.zeros(2048)
    for x1,x2 in _fiber1:
        fiber1_template[x1+10:x2-10] = 1
    fiber2_template = np.zeros(2048)
    for x1,x2 in _fiber2:
        fiber2_template[x1+10:x2-10] = 1
    fiber3_template = np.zeros(2048)
    for x1,x2 in _fiber3:
        fiber3_template[x1+10:x2-10] = 1
    fiber4_template = np.zeros(2048)
    for x1,x2 in _fiber4:
        fiber4_template[x1+10:x2-10] = 1
    flattened = np.nanmean(image,axis=1)

    # import matplotlib.pyplot as plt
    # plt.plot(flattened/np.nanmax(flattened),label="flat")
    # # plt.plot(fiber1_template,label="0")
    # plt.plot(fiber2_template,label="1")
    # # plt.plot(fiber3_template,label="2")
    # # plt.plot(fiber4_template,label="3")
    # plt.show()

    return np.argmax([np.nansum(fiber1_template * flattened),
                       np.nansum(fiber2_template * flattened),
                       np.nansum(fiber3_template * flattened),
                       np.nansum(fiber4_template * flattened)])