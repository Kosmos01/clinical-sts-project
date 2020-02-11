import string
import random




def getScore(rowNumber):

    trainingExamplesFile = open('scoresWithConcepts.txt', 'a+')
    rowIter=1
    with open('sentenceScores.txt', 'r') as r:
        for line in r:
            if rowIter == rowNumber:


    totalNumber = (zeros + ones + twos + threes + fours + fives)

    for t in range(0, 5):
        random.shuffle(fivePairs)
        random.shuffle(fourPairs)
        random.shuffle(threePairs)
        random.shuffle(twoPairs)
        random.shuffle(onePairs)
        random.shuffle(zeroPairs)

    trainingExamplesFile.write(
        '25 pairs from each category have been collected (random shuffle) and listed, but the totals of each category are displayed below:\n')
    trainingExamplesFile.write(
        'Sums -> Total: {}, Zeroes: {}, Ones: {}, Twos: {}, Threes: {}, Fours: {}, Fives: {}\n'.format(totalNumber,
                                                                                                       zeros,
                                                                                                       ones, twos,
                                                                                                       threes,
                                                                                                       fours, fives))
    trainingExamplesFile.write(
        "{}% of all training data have been 'hard' scored ({}/{})\n\n".format((totalNumber / 1642) * 100, totalNumber,
                                                                              1642))

    trainingExamplesFile.write(
        'Score 5 -> The two sentences are completely equivalent, as they mean the same thing.\n\n')
    for x in range(0, 25):
        trainingExamplesFile.write(str(x + 1) + ') ' + fivePairs[x])
    trainingExamplesFile.write('\n')

    trainingExamplesFile.write(
        'Score 4 -> The two sentences are mostly equivalent, but some unimportant details differ.\n\n')
    for x in range(0, 25):
        trainingExamplesFile.write(str(x + 1) + ') ' + fourPairs[x])
    trainingExamplesFile.write('\n')

    trainingExamplesFile.write(
        'Score 3 -> The two sentences are roughly equivalent, but some important information differs/missing.\n\n')
    for x in range(0, 25):
        trainingExamplesFile.write(str(x + 1) + ') ' + threePairs[x])
    trainingExamplesFile.write('\n')

    trainingExamplesFile.write('Score 2 -> The two sentences are not equivalent, but share some details.\n\n')
    for x in range(0, 25):
        trainingExamplesFile.write(str(x + 1) + ') ' + twoPairs[x])
    trainingExamplesFile.write('\n')

    trainingExamplesFile.write('Score 1 -> The two sentences are not equivalent, but are on the same topic\n\n')
    for x in range(0, 25):
        trainingExamplesFile.write(str(x + 1) + ') ' + onePairs[x])
    trainingExamplesFile.write('\n')

    trainingExamplesFile.write('Score 0 -> The two sentences are completely dissimilar.\n\n')
    for x in range(0, 25):
        trainingExamplesFile.write(str(x + 1) + ') ' + zeroPairs[x])

    totalNumber = (zeros + ones + twos + threes + fours + fives)
    print(
        'Sums -> Total: {}, Zeroes: {}, Ones: {}, Twos: {}, Threes: {}, Fours: {}, Fives: {}'.format(totalNumber, zeros,
                                                                                                     ones, twos, threes,
                                                                                                     fours, fives))


def main():


if __name__ == '__main__':
    main()