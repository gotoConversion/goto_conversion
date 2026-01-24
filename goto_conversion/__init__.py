def convertAmericanOdds(listOfOdds):
    try: #using numpy
        import numpy as np
        listOfOdds = listOfOdds.astype(float)
        isNegativeAmericanOdds = listOfOdds < 0.0
        listOfOdds[isNegativeAmericanOdds] = 1.0 + ((100.0 / listOfOdds[isNegativeAmericanOdds]) * -1.0)
        listOfOdds[~isNegativeAmericanOdds] = 1.0 + (listOfOdds[~isNegativeAmericanOdds] / 100.0)
    except: #using base python
        for i in range(len(listOfOdds)):
            currOdds = listOfOdds[i]
            isNegativeAmericanOdds = currOdds < 0.0
            if isNegativeAmericanOdds:
                currDecimalOdds = 1.0 + (100.0/(currOdds*-1.0))
            else: #Is non-negative American Odds
                currDecimalOdds = 1.0 + (currOdds/100.0)
            listOfOdds[i] = currDecimalOdds
    return listOfOdds

def errorCatchers(listOfOdds):
    if len(listOfOdds) < 2:
        raise ValueError('len(listOfOdds) must be >= 2')
    try:
        import numpy as np
        isAllOddsAbove1 = np.all(listOfOdds > 1.0)
    except:
        isAllOddsAbove1 = all([x > 1.0 for x in listOfOdds])
    if not isAllOddsAbove1:
        raise ValueError('All odds must be > 1.0, set isAmericanOdds parameter to True if using American Odds')

def efficient_shin_conversion(listOfOdds, total = 1.0, multiplicativeIfImprudentOdds = False, isAmericanOdds = False):

    #Convert American Odds to Decimal Odds
    if isAmericanOdds:
        listOfOdds = convertAmericanOdds(listOfOdds)

    #Error Catchers
    errorCatchers(listOfOdds)

    try: #using numpy
        import numpy as np
        #Compute parameters
        listOfPies = 1.0 / listOfOdds
        beta = np.sum(listOfPies)
        listOfComplementPies = listOfPies - (beta - listOfPies)

        #Compute vectors
        listOfZ = ((beta - 1.0) * (listOfComplementPies ** 2.0 - beta)) / (beta * (listOfComplementPies ** 2.0 - 1.0))
        listOfPStars = ((np.sqrt(listOfZ**2.0 + 4.0 * (1.0 - listOfZ) * (listOfPies**2 / beta)) - listOfZ) / (2.0 * (1.0 - listOfZ)))
        normalizer = np.sum(listOfPStars) / total
        outputListOfProbabilities = listOfPStars / normalizer

    except: #using base python
        #Compute parameters
        listOfPies = [1.0/x for x in listOfOdds]
        beta = sum(listOfPies)
        listOfComplementPies = [x - (beta-x) for x in listOfPies]

        #Compute vectors
        listOfZ = [((beta - 1.0)*(x**2.0 - beta))/(beta*(x**2.0 - 1.0)) for x in listOfComplementPies]
        listOfPStars = [(((z_i**2.0 + 4.0*(1.0-z_i)*(pi_i**2.0/beta))**0.5) - z_i)/(2.0*(1.0 - z_i)) for pi_i,z_i in zip(listOfPies, listOfZ)]
        normalizer = sum(listOfPStars)/total
        outputListOfProbabilities = [x/normalizer for x in listOfPStars]

    return outputListOfProbabilities

def goto_conversion(listOfOdds, total = 1.0, multiplicativeIfImprudentOdds = False, isAmericanOdds = False):

    #Convert American Odds to Decimal Odds
    if isAmericanOdds:
        listOfOdds = convertAmericanOdds(listOfOdds)

    #Error Catchers
    errorCatchers(listOfOdds)

    try: #using numpy
        import numpy as np
        listOfProbabilities = 1.0 / listOfOdds
        listOfSe = np.sqrt((listOfProbabilities - listOfProbabilities**2.0) / listOfProbabilities)
        step = (np.sum(listOfProbabilities) - total) / np.sum(listOfSe)
        outputListOfProbabilities = listOfProbabilities - (listOfSe * step)
        if np.any(outputListOfProbabilities <= 0.0) or (np.sum(listOfProbabilities) <= 1.0):
            if multiplicativeIfImprudentOdds:
                normalizer = np.sum(listOfProbabilities) / total
                outputListOfProbabilities = np.array(listOfProbabilities) / normalizer
            else:
                print('Odds must have a positive low bookmaker margin to be prudent.')
                raise ValueError('Set multiplicativeIfImprudentOdds argument to True to use multiplicative conversion for Imprudent odds.')

    except: #using base python
        listOfProbabilities = [1.0/x for x in listOfOdds] #initialize probabilities using inverse odds
        listOfSe = [pow((x-x**2.0)/x,0.5) for x in listOfProbabilities] #compute the standard error (SE) for each probability
        step = (sum(listOfProbabilities) - total)/sum(listOfSe) #compute how many steps of SE the probabilities should step back by
        outputListOfProbabilities = [x - (y*step) for x,y in zip(listOfProbabilities, listOfSe)]
        if any(0.0 >= x for x in outputListOfProbabilities) or (sum(listOfProbabilities) <= 1.0):
            if multiplicativeIfImprudentOdds:
                normalizer = sum(listOfProbabilities)/total
                outputListOfProbabilities = [x/normalizer for x in listOfProbabilities]
            else:
                print('Odds must have a positive low bookmaker margin to be prudent.')
                raise ValueError('Set multiplicativeIfImprudentOdds argument to True to use multiplicative conversion for Imprudent odds.')

    return outputListOfProbabilities

def zero_sum(listOfPrices, listOfVolumes):
    listOfSe = [x**0.5 for x in listOfVolumes] #compute standard errors assuming standard deviation is same for all stocks
    step = sum(listOfPrices)/sum(listOfSe)
    outputListOfPrices = [x - (y*step) for x,y in zip(listOfPrices, listOfSe)]
    return outputListOfPrices

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import os

def pgd_attack(model, images, labels, eps, alpha, steps):
    """
    Projected Gradient Descent (PGD) - The "FakeIt" approach.
    """
    adv_images = images.clone().detach()

    # Random initialization within the epsilon ball
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, 0, 1)

    loss_fn = nn.CrossEntropyLoss()

    for i in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        # Maximize loss for the "Fake" label (move away from being detected as fake)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv_images = adv_images.detach() + alpha * adv_images.grad.sign()

        # Projection
        eta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + eta, min=0, max=1)

    return adv_images

def image_conversion(image_path, output_filename, eps=0.03, alpha=0.01, steps=40):
    # 1. Setup Model
    detector = models.resnet50(pretrained=True)
    detector.eval()

    # 2. Check Input
    #image_path = "fake.png"
    if not os.path.exists(image_path):
        # Create a dummy image if one doesn't exist for testing purposes
        print(f"'{image_path}' not found. Creating a dummy test image...")
        dummy = Image.new('RGB', (400, 300), color = 'red')
        dummy.save(image_path)

    # 3. Prepare Transform
    # MODIFIED: Removed transforms.Resize((224, 224)) to preserve aspect ratio.
    # The model will now process the image at its original resolution.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0)

    print(f"Processing image with size: {input_image.size}")

    # Define True Label (1 = Fake)
    true_label = torch.tensor([1])

    print(f"Running PGD Attack on {image_path}...")
    adv_example = pgd_attack(detector, input_tensor, true_label, eps, alpha, steps)

    # Check results
    old_pred = detector(input_tensor).argmax(1).item()
    new_pred = detector(adv_example).argmax(1).item()

    print(f"Prediction changed from {old_pred} to {new_pred}")

    # Save Output
    #output_filename = "adv_pgd_output.png"
    transforms.ToPILImage()(adv_example.squeeze(0)).save(output_filename)
    print(f"Saved adversarial image to '{output_filename}' with original aspect ratio.")
