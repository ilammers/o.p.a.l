import time
from library import *
from multiprocessing import Process

# main exploit file used once the unit is deployed

pf = 20.0 # base sine frequency, Hz
hz1 = 0.0 # modified sine frequency, Hz
hz2 = 0.0 # modified sine frequency, Hz

mod1 = 1.25 # sine modifer for hz1
mod2 = 1.5 # sine modifer for hz2

iterable = 5 # defines number of perturbation generated

for i in range(iterable):
    # increase hz every pass, multiply base by modifier to raise chord proportionally
    hz1 += pf * mod1
    hz2 += pf * mod2

    filename = 'adv' + str(i) + '-' + time.strftime("%Y%m%d-%H%M%S")

    if i == 0:
        print('loop index is ' + str(i))
        capture('initial')
        predict('/home/pi/initial.jpg')
    else:
        # this creates a race condition depending on the value given to duration in generateSinewave()
        # duration must equal to or exeed the time it takes capture to execute
        # multiprocessing is used to run these function synchronously
        wave_proc = Process(target = generateSinewave, args = (hz1, hz2))
        cap_proc = Process(target = capture, args = (filename,))
        
        # start and sync processes
        wave_proc.start()
        cap_proc.start()
        wave_proc.join()
        cap_proc.join()

        # kill process for next pass
        wave_proc.terminate()
        cap_proc.terminate()

        # get model prediction
        predict(filename + '.jpg')
