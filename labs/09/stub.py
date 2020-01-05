import numpy as np
import az_quiz
#
# count = 0
# for j in range(7):
#     for i in range(j + 1):
#         count += 1
#         # print(str(j) + " " + str(i))
#         print(i ,j)
#
# print (count)

def softmax_sample(x):
  """Compute softmax values for each sets of scores in x."""
  return np.exp(x) / np.sum(np.exp(x), axis=0)

# print(list(map(softmax_sample, list(range(4)))))
# l = list(range(8)) + [10]
# print(l)
# print(softmax_sample(l))

# board = np.zeros([self.N, self.N], dtype=np.uint8)
game = az_quiz.AZQuiz(False)
# print(list(range(28)))
game.move(1)
any_anchor = any(map(game.valid, list(range(28))))

print(any_anchor)

print(list(map(game.valid, list(range(28)))))


