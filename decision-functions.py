import random
import numpy
import math

n = 0
r = 0
R = 0
t = 0
alpha = []
l_ratio = []
drop = 0
STAGE_2 = 18
STAGE_3 = 60
STAGE_4 = 100

def hunt_choices(round_number, current_food, current_reputation, m,  player_reputations):
    
    global n
    global r
    global t
    global R
    
    t = round_number
    if current_reputation == 0:
        hunt_decisions = []
        n = int(len(player_reputations) * random.random())
        for _ in range(n):
            hunt_decisions.append('h')
        while len(hunt_decisions) < len(player_reputations):
            hunt_decisions.append('s')
    else:
        R = list(player_reputations)
        R.append(current_reputation)
        r = current_reputation
        p = len(R)
        mu = p * numpy.mean(R) * 1.0 / r
        sd = numpy.sqrt(p) * numpy.std(R) * 1.0 / r
        if(len(l_ratio) > 2):
            mu = ((len(l_ratio) - 1) * numpy.mean(l_ratio) + (t + 1 - len(l_ratio)) * mu) / t
            sd = ((len(l_ratio) - 1) * numpy.std(l_ratio) + (t + 1 - len(l_ratio)) * sd) / t
        nvec = range(p - 1)
        nvec = [(mu - (m / (k + 1))) / (sd * numpy.sqrt(2)) for k in nvec]
        for i in range(p - 1):
            nvec[i] = (p - 1) * (1 + erf(nvec[i])) - (i + 1)
        n = numpy.argmax(nvec)
        
        if t >= STAGE_2:
            if(len(alpha) != 0):
                rvec = []
                alpha_mean = numpy.mean(alpha)
                for x in range(p):
                    r_i = (x + 1) / p
                    probvec = range(p - 1)
                    probvec = [(mu - (p * R[k] * (1 - r_i) * 1.0 / (r_i * r_i))) / (sd * numpy.sqrt(2)) for k in probvec]
                    probvec = [alpha_mean * (1 + erf(k)) for k in probvec]
                    rvec.append(numpy.sum(probvec) + (p - 1) * (1 + erf((mu - (m * 1.0 / n)) / (sd * numpy.sqrt(2)))))
                r_z = (numpy.argmax(rvec) + 1)
                if t >= STAGE_4: n = r_z
                else:
                    r_z = r_z / p
                    ep = (t + 1) * (p - 1) * (r_z - (t * r * 1.0 / (t + 1))) - n
                    if ep < 0: ep = -1 * numpy.sqrt(-1 * ep)
                    else: ep = numpy.sqrt(ep)
                    n += ep
                n = int(n)
                if(n < 0): n = 0
                if(n > p - 1): n = p - 1
            
        index = [int(j) for j in numpy.argsort(R).tolist()]
        hunt_decisions = ['s'] * len(player_reputations)
        if t >= STAGE_3:
            for _ in range(n):
                z = index.pop()
                if z > drop: hunt_decisions[z] = 'h'
        else:
            for _ in range(n):
                hunt_decisions[index.pop()] = 'h'

    return hunt_decisions

def hunt_outcomes(food_earnings):
    
    global alpha
    global drop
    
    if r != 0:
        p = len(R)
        mu = p * numpy.mean(R) * 1.0 / r
        sd = numpy.sqrt(p) * numpy.std(R) * 1.0 / r
        if(len(l_ratio) > 2):
            mu = ((len(l_ratio) - 1) * numpy.mean(l_ratio) + (t + 1 - len(l_ratio)) * mu) / t
            sd = ((len(l_ratio) - 1) * numpy.std(l_ratio) + (t + 1 - len(l_ratio)) * sd) / t
        H = numpy.sum(food_earnings) + n + 2 * len(food_earnings)
        probvec = range(len(food_earnings))
        probvec = [(mu - (p * R[k] * (1 - r) * 1.0 / (r * r))) / (sd * numpy.sqrt(2)) for k in probvec]
        probvec = [1 + erf(k) for k in probvec]
        alpha.append(H / numpy.sum(probvec))
        
    if t >= (STAGE_3 - 1):
        dropx = []
        for i in range(len(food_earnings)):
            if food_earnings[i] == -3:
                dropx.append(R[i])
        if len(dropx) != 0:
            drop = numpy.mean(dropx)
    

def round_end(award, m, number_hunters):
    
    global l_ratio
    
    if n != 0: l_ratio.append(number_hunters * 1.0 / n)
    if len(l_ratio) == 2:
        l_ratio[0] = l_ratio[1]

def erf(x):
    #This implementation of the error function comes from "Handbook of Mathematical Functions", formula 7.1.26
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    return sign*y