def goto_conversion(listOfOdds, total = 1, eps = 1e-6, isAmericanOdds = False):

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
    listOfMoe = [pow((x-x**2)/x,0.5) for x in listOfProbabilities] #compute the margin of error (MOE) for each probability
    step = (sum(listOfProbabilities) - total)/sum(listOfMoe) #compute how many steps of MOE the probabilities should step back by
    outputListOfProbabilities = [min(max(x - (y*step),eps),1) for x,y in zip(listOfProbabilities, listOfMoe)]
    return outputListOfProbabilities
