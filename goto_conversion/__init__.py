def goto_conversion(listOfOdds, total = 1, multiplicativeIfUnprudentOdds = False, isAmericanOdds = False):

    #Convert American Odds to Decimal Odds
    if isAmericanOdds:
        for i in range(len(listOfOdds)):
            currOdds = listOfOdds[i]
            isNegativeAmericanOdds = currOdds < 0
            if isNegativeAmericanOdds:
                currDecimalOdds = 1 + (100/(currOdds*-1))
            else: #Is non-negative American Odds
                currDecimalOdds = 1 + (currOdds/100)
            listOfOdds[i] = currDecimalOdds

    #Error Catchers
    if len(listOfOdds) < 2:
        raise ValueError('len(listOfOdds) must be >= 2')
    if any(x < 1 for x in listOfOdds):
        raise ValueError('All odds must be >= 1, set isAmericanOdds parameter to True if using American Odds')

    #Computation
    listOfProbabilities = [1/x for x in listOfOdds] #initialize probabilities using inverse odds
    listOfSe = [pow((x-x**2)/x,0.5) for x in listOfProbabilities] #compute the standard error (SE) for each probability
    step = (sum(listOfProbabilities) - total)/sum(listOfSe) #compute how many steps of SE the probabilities should step back by
    outputListOfProbabilities = [x - (y*step) for x,y in zip(listOfProbabilities, listOfSe)]
    if any(0 >= x for x in outputListOfProbabilities) or (sum(listOfProbabilities) <= 1):
        if multiplicativeIfUnprudentOdds:
            normalizer = sum(listOfProbabilities)/total
            outputListOfProbabilities = [x/normalizer for x in listOfProbabilities]
        else:
            print('Odds must have a positive low bookmaker margin to be prudent.')
            raise ValueError('Set multiplicativeIfUnprudentOdds argument to True to use multiplicative conversion for unrpudent odds.')
    return outputListOfProbabilities

def zero_sum(listOfPrices, listOfVolumes):
    listOfSe = [x**0.5 for x in listOfVolumes] #compute standard errors assuming standard deviation is same for all stocks
    step = sum(listOfPrices)/sum(listOfSe)
    outputListOfPrices = [x - (y*step) for x,y in zip(listOfPrices, listOfSe)]
    return outputListOfPrices
