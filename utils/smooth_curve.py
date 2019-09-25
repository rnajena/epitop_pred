import numpy as np
import matplotlib.pyplot as plt

pwm_frame_extend = 5
scores = [1,2,5,3,6,2,4,6,2,5,3,5,5,6,8,2,3,3,3,2,0,2,0,0,0,2,2,2,1,4,6,5,6,]
seqlen = len(scores)

# calculate positon weight matrix

weight = pwm_frame_extend + 1
weights = []
for i in range(pwm_frame_extend+1):
   weights.append(i)
weights = weights[pwm_frame_extend:0:-1] + weights
weights = [(weight -x)/(weight) for x in weights]

# pwm = []
# pwm_adapted =[]
# for i in range(seqlen):
#    out = []
#    for j in range(len(weights)):
#       if i + j - pwm_frame_extend >= 0 and j - pwm_frame_extend + i < seqlen:
#          out.append(weights[j])
#    if len(out) < seqlen:
#       if seqlen - i - len(out) > 0:
#          if (i-pwm_frame_extend) < 0:
#             out = out + [0] * (seqlen - len(out))
#          else:
#             out = out + [0] * (seqlen - len(out)-(i-pwm_frame_extend))
#
#       if len(out) < seqlen:
#          out = [0] * (seqlen - len(out)) + out
#    pwm.append(out)
# pwm = np.zeros((seqlen,seqlen))
# for index in range(seqlen):
#     pwm[index][index-pwm_frame_extend:index+pwm_frame_extend] += 1 #np.array(weights)

pwm_adapted = pwm.copy()
# multiply scores to pwm
for i in range(len(scores)):
   pwm_adapted[i] = [scores[i] * pwm_score for pwm_score in pwm[i]]
# sum up and normalize per position
pwm_adapted = np.array(pwm_adapted)
pwm = np.array(pwm)
pwm_scores = np.array(pwm_adapted).sum(axis=0) / np.array(pwm).sum(axis=1)
plt.plot(range(seqlen), pwm_scores)
plt.plot(range(seqlen), scores)
plt.plot(range(seqlen), np.array(pwm).sum(axis=1))
plt.show()