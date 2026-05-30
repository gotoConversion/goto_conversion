import os
import logging

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    import torchvision.transforms as transforms
    from PIL import Image
    import nltk
    import requests
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, ViTForImageClassification, ViTImageProcessor
except ImportError:
    pass


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

class AdversarialImageConverter:
    """
    Converts AI-generated images to bypass AI detectors using adversarial
    perturbations against a local ViT-based AI image detection model,
    validated by the AI or Not API.
    """

    def __init__(self, detector_name="dima806/ai_vs_real_image_detection"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading Image Detector: {detector_name}...")
        self.processor = ViTImageProcessor.from_pretrained(detector_name)
        self.model = ViTForImageClassification.from_pretrained(detector_name)
        self.model.to(self.device)
        self.model.eval()

        self.label_map = self.model.config.id2label
        self.real_idx = None
        self.fake_idx = None
        for idx, label in self.label_map.items():
            if label.upper() in ("REAL", "HUMAN"):
                self.real_idx = int(idx)
            elif label.upper() in ("FAKE", "AI-GENERATED", "AI"):
                self.fake_idx = int(idx)

    def get_local_score(self, raw_pixels):
        """
        Returns probability of image being REAL according to local detector.
        raw_pixels: tensor of shape (1, 3, 224, 224) in [0,1] range.
        """
        normalized = self._normalize(raw_pixels)
        with torch.no_grad():
            outputs = self.model(pixel_values=normalized)
            probs = torch.softmax(outputs.logits, dim=1)
        return probs[0, self.real_idx].item()

    def check_aiornot(self, image_path, api_token):
        """
        Checks image classification via AI or Not API.
        Returns dict with 'verdict', 'ai_confidence', 'human_confidence'.
        """
        url = "https://api.aiornot.com/v2/image/sync"
        headers = {"Authorization": f"Bearer {api_token}"}

        with open(image_path, "rb") as f:
            resp = requests.post(
                url, headers=headers,
                files={"image": f},
                params={"only": "ai_generated"}
            )

        if resp.status_code != 200:
            print(f"  API Error: {resp.status_code} - {resp.text}")
            return None

        data = resp.json()
        report = data["report"]["ai_generated"]
        return {
            "verdict": report["verdict"],
            "ai_confidence": report["ai"]["confidence"],
            "human_confidence": report["human"]["confidence"]
        }

    def _normalize(self, tensor):
        """
        Applies ViT ImageNet normalization to a [0,1] tensor.
        """
        mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def pgd_attack(self, raw_pixels, eps, alpha, steps):
        """
        PGD attack in raw [0,1] pixel space. Normalization is applied
        inside each forward pass to preserve correct image representation.
        """
        original = raw_pixels.clone().detach()
        adv = raw_pixels.clone().detach()
        adv = adv + torch.empty_like(adv).uniform_(-eps * 0.1, eps * 0.1)
        adv = torch.clamp(adv, 0, 1)

        target_label = torch.tensor([self.real_idx], device=self.device)
        loss_fn = nn.CrossEntropyLoss()

        for i in range(steps):
            adv.requires_grad = True
            normalized = self._normalize(adv)
            outputs = self.model(pixel_values=normalized)

            # Minimize loss for REAL label (maximize REAL probability)
            loss = loss_fn(outputs.logits, target_label)

            self.model.zero_grad()
            loss.backward()

            adv = adv.detach() - alpha * adv.grad.sign()

            # Projection in [0,1] space
            eta = torch.clamp(adv - original, min=-eps, max=eps)
            adv = torch.clamp(original + eta, min=0, max=1)

        return adv.detach()

    def convert(self, image_path, output_path, api_token,
                max_retries=5, initial_eps=0.02, initial_steps=50):
        """
        Iteratively perturbs the image until AI or Not classifies it as human.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        original_size = img.size
        print(f"Processing image: {image_path} (size: {original_size})")

        # Initial API check
        api_result = self.check_aiornot(image_path, api_token)
        if api_result:
            print(f"Initial AI or Not: verdict={api_result['verdict']}, "
                  f"AI={api_result['ai_confidence']:.4f}, "
                  f"Human={api_result['human_confidence']:.4f}")
            if api_result["verdict"] == "human":
                print("Image already classified as human. No conversion needed.")
                img.save(output_path)
                return output_path

        # Resize to 224x224 and convert to [0,1] tensor
        img_resized = img.resize((224, 224), Image.LANCZOS)
        raw_pixels = transforms.ToTensor()(img_resized).unsqueeze(0).to(self.device)

        local_score = self.get_local_score(raw_pixels)
        print(f"Initial local REAL score: {local_score:.4f}")

        best_pixels = raw_pixels.clone()
        best_score = local_score

        for attempt in range(max_retries):
            eps = initial_eps * (1.0 + attempt * 0.5)
            alpha = eps / 10.0
            steps = initial_steps + (attempt * 25)

            print(f"\n=== Pass {attempt + 1}/{max_retries} "
                  f"(eps={eps:.4f}, alpha={alpha:.5f}, steps={steps}) ===")

            # Run PGD attack in [0,1] pixel space
            adv_pixels = self.pgd_attack(
                best_pixels, eps=eps, alpha=alpha, steps=steps
            )

            new_score = self.get_local_score(adv_pixels)
            print(f"  Local REAL score after PGD: {new_score:.4f}")

            if new_score > best_score:
                best_pixels = adv_pixels
                best_score = new_score

            # Save best result for API check
            best_img_224 = transforms.ToPILImage()(best_pixels.squeeze(0).cpu())
            best_img_full = best_img_224.resize(original_size, Image.LANCZOS)
            best_img_full.save(output_path, quality=95)

            # Validate with AI or Not API
            api_result = self.check_aiornot(output_path, api_token)
            if api_result:
                print(f"  AI or Not: verdict={api_result['verdict']}, "
                      f"AI={api_result['ai_confidence']:.4f}, "
                      f"Human={api_result['human_confidence']:.4f}")

                if api_result["verdict"] == "human":
                    print("\nSuccess: AI or Not classifies image as HUMAN!")
                    return output_path

        print(f"\nWarning: max_retries ({max_retries}) exceeded. "
              f"Returning best result (local score: {best_score:.4f}).")
        return output_path


def image_conversion(image_path, output_filename, aiornot_token,
                     max_retries=5, initial_eps=0.02, initial_steps=50):
    """
    Converts an AI-generated image to bypass the AI or Not detector using
    adversarial perturbations against a local ViT-based AI detection model,
    validated by the AI or Not API.

    Args:
        image_path: Path to the input AI-generated image
        output_filename: Path to save the converted image
        aiornot_token: AI or Not API token for validation
        max_retries: Maximum number of adversarial passes
        initial_eps: Initial perturbation budget (increases per pass)
        initial_steps: Initial PGD steps (increases per pass)

    Returns:
        Path to the output image
    """
    logging.getLogger("transformers").setLevel(logging.ERROR)

    converter = AdversarialImageConverter()
    result = converter.convert(
        image_path, output_filename, aiornot_token,
        max_retries=max_retries,
        initial_eps=initial_eps,
        initial_steps=initial_steps
    )

    print("-" * 50)
    print(f"Output saved to: {result}")
    print("-" * 50)
    return result

class AdversarialParaphraser:
    def __init__(self,
                 detector_name="Hello-SimpleAI/chatgpt-detector-roberta",
                 paraphraser_name="Vamsi/T5_Paraphrase_Paws"):
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
        """
        Generates variations of a single sentence.
        """
        text = "paraphrase: " + sentence + " </s>"
        encoding = self.p_tokenizer(text, padding="longest", return_tensors="pt")
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



class AdversarialParaphraserEnhanced(AdversarialParaphraser):
    """
    Enhanced paraphraser using humarin/chatgpt_paraphraser_on_T5_base for
    higher-quality, more diverse paraphrases with a multi-pass strategy.
    """

    def __init__(self,
                 detector_name="Hello-SimpleAI/chatgpt-detector-roberta",
                 paraphraser_name="humarin/chatgpt_paraphraser_on_T5_base"):
        super().__init__(detector_name=detector_name, paraphraser_name=paraphraser_name)

    def convert(self, text, goal_threshold=0.95, max_retries=5):
        print(f"\nOriginal Text Start: {text[:80]}...")

        current_prob = self.get_probability(text, target_label_idx=0)
        print(f"Initial 'Real' Probability: {current_prob:.4f}")

        if current_prob > goal_threshold:
            print("Text is already detected as Real.")
            return text

        best_text = text
        best_prob = current_prob

        for outer_pass in range(max_retries):
            num_candidates = 8 + (outer_pass * 4)
            min_accept_threshold = max(0.5, best_prob - 0.05) if outer_pass > 0 else best_prob
            print(f"\n=== Pass {outer_pass + 1}/{max_retries} "
                  f"(candidates: {num_candidates}, "
                  f"accept threshold: {min_accept_threshold:.4f}) ===")

            try:
                sentences = nltk.sent_tokenize(best_text)
            except Exception:
                sentences = best_text.split('. ')

            for i, original_sent in enumerate(sentences):
                if len(original_sent) < 10:
                    continue

                print(f"  Processing Sentence {i+1}/{len(sentences)}: "
                      f"'{original_sent[:50]}...'")

                candidates = self.generate_paraphrases(
                    original_sent, num_return_sequences=num_candidates
                )

                local_best_sent = original_sent
                local_best_prob = best_prob
                sent_improved = False

                for candidate in candidates:
                    temp_sentences = sentences[:]
                    temp_sentences[i] = candidate
                    temp_full_text = " ".join(temp_sentences)

                    prob = self.get_probability(temp_full_text, target_label_idx=0)

                    if outer_pass == 0:
                        if prob > best_prob:
                            best_prob = prob
                            best_text = temp_full_text
                            local_best_sent = candidate
                            local_best_prob = prob
                            sent_improved = True
                    else:
                        if prob >= min_accept_threshold and candidate != original_sent:
                            if prob > local_best_prob or not sent_improved:
                                local_best_prob = prob
                                local_best_sent = candidate
                                sent_improved = True

                if sent_improved:
                    sentences[i] = local_best_sent
                    best_text = " ".join(sentences)
                    best_prob = self.get_probability(best_text, target_label_idx=0)
                    print(f"    -> {'Improved' if outer_pass == 0 else 'Diversified'}! "
                          f"Score: {best_prob:.4f}")

            print(f"\n  After pass {outer_pass + 1}: Score = {best_prob:.4f}")

            if best_prob > goal_threshold:
                print("Success: Target threshold reached!")
                return best_text

        print(f"Warning: max_retries ({max_retries}) exceeded. "
              f"Returning best text achieved (score: {best_prob:.4f}).")
        return best_text


def text_conversion(ai_text):
    """
    Converts AI-generated text to bypass AI detectors using multi-pass
    adversarial paraphrasing with the humarin/chatgpt_paraphraser_on_T5_base model.
    """
    # Configure logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Download nltk sentence tokenizer data
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    adversary = AdversarialParaphraserEnhanced()

    final_text = adversary.convert(ai_text)

    print("-" * 50)
    print("Final Output:")
    print(final_text)
    print("-" * 50)
    return final_text
