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

def pgd_attack(model, images, labels, eps, alpha, steps):
    """
    Projected Gradient Descent (PGD) - The "FakeIt" approach.
    """
    import torch
    import torch.nn as nn
    from torchvision import models
    import torchvision.transforms as transforms
    from PIL import Image
    import os
    
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
    # 0. Import Libraries
    import torch
    import torch.nn as nn
    from torchvision import models
    import torchvision.transforms as transforms
    from PIL import Image
    import os
    
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

class AdversarialParaphraser:
    def __init__(self,
                 detector_name="Hello-SimpleAI/chatgpt-detector-roberta",
                 paraphraser_name="Vamsi/T5_Paraphrase_Paws"):

        import torch
        import nltk
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
        import logging
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Load Detector (Judge) ---
        print(f"Loading Detector: {detector_name}...")
        self.d_tokenizer = AutoTokenizer.from_pretrained(detector_name)
        self.d_model = AutoModelForSequenceClassification.from_pretrained(detector_name)
        self.d_model.to(self.device)
        self.d_model.eval()

        # --- Load Paraphraser (Generator) ---
        print(f"Loading Paraphraser: {paraphraser_name}...")
        self.p_tokenizer = AutoTokenizer.from_pretrained(paraphraser_name)
        self.p_model = AutoModelForSeq2SeqLM.from_pretrained(paraphraser_name)
        self.p_model.to(self.device)
        self.p_model.eval()

    def get_probability(self, text, target_label_idx=0):
        import torch
        import nltk
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
        import logging
        """
        Returns probability of text being Human (Index 0).
        """
        if not text: return 0.0
        inputs = self.d_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.d_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        return probs[0, target_label_idx].item()

    def generate_paraphrases(self, sentence, num_return_sequences=5):
        import torch
        import nltk
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
        import logging
        """
        Generates variations of a single sentence.
        """
        text = "paraphrase: " + sentence + " </s>"
        encoding = self.p_tokenizer.encode_plus(text, padding="longest", return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_masks = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.p_model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                max_length=256,
                do_sample=True,          # Add randomness
                top_k=120,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=num_return_sequences
            )

        res = [self.p_tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
               for output in outputs]
        return set(res) # Return unique paraphrases only

    def convert(self, text, goal_threshold=0.95):
        import torch
        import nltk
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
        import logging
        print(f"\nOriginal Text Start: {text[:80]}...")

        # Initial Check
        current_prob = self.get_probability(text, target_label_idx=0)
        print(f"Initial 'Real' Probability: {current_prob:.4f}")

        if current_prob > 0.5:
            print("Text is already detected as Real.")
            return text

        # Split text into sentences
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback if nltk fails
            sentences = text.split('. ')

        best_text = text
        best_prob = current_prob

        # Iterate through each sentence
        for i, original_sent in enumerate(sentences):
            if len(original_sent) < 10: continue # Skip tiny fragments

            print(f"\nProcessing Sentence {i+1}/{len(sentences)}: '{original_sent[:50]}...'")

            # Generate candidates
            candidates = self.generate_paraphrases(original_sent, num_return_sequences=4)

            sent_improved = False
            local_best_sent = original_sent

            for candidate in candidates:
                # Construct the full paragraph with this candidate
                # (Create a copy of the list to modify)
                temp_sentences = sentences[:]
                temp_sentences[i] = candidate
                temp_full_text = " ".join(temp_sentences) # Join with spaces

                # Check score
                prob = self.get_probability(temp_full_text, target_label_idx=0)

                # If this candidate improves the overall score, keep it
                if prob > best_prob:
                    best_prob = prob
                    best_text = temp_full_text
                    local_best_sent = candidate
                    sent_improved = True

            if sent_improved:
                print(f"  -> Improved! New Score: {best_prob:.4f}")
                print(f"  -> Swapped to: '{local_best_sent}'")
                sentences[i] = local_best_sent # Update the main list for subsequent steps
            else:
                print("  -> No improvement found for this sentence.")

            if best_prob > goal_threshold:
                print("\nSuccess: Target threshold reached!")
                break

        return best_text

# --- Usage ---
#if __name__ == "__main__":
def text_conversion(ai_text):
    import torch
    import nltk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
    import logging

    # Configure logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Download nltk sentence tokenizer data
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    adversary = AdversarialParaphraser()

    #ai_text = "In a rare weekend of market activity, global equities are navigating a risk-off environment as investors digest a hawkish shift in U.S. monetary expectations following the nomination of Kevin Warsh as the next Federal Reserve Chair. On Friday, Wall Street closed the week in the red, with the S&P 500 dropping 0.52% and the Dow Jones sliding 0.85%, while tech stocks took a hit as the Nasdaq fell 0.66% on fears of a slower-than-expected easing cycle. This sentiment spilled into a historic Sunday trading session in India, where benchmarks opened with cautious volatility as Finance Minister Nirmala Sitharaman presented the Union Budget 2026, aiming for a 7.4% growth target amid global uncertainty. Meanwhile, precious metals faced a significant correction, with gold and silver prices retreating sharply from recent highs as the U.S. dollar strengthened, leaving international traders bracing for a potentially turbulent week ahead as central bank independence and corporate earnings remain in the spotlight."

    final_text = adversary.convert(ai_text)

    print("-" * 50)
    print("Final Output:")
    print(final_text)
    print("-" * 50)
    return final_text
