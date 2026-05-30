import os
import logging

try:
    from PIL import Image, ImageFilter
except ImportError:
    pass

try:
    import torch
    import nltk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
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
    Converts AI-generated images to bypass AI detectors using
    image processing techniques.
    """

    def convert(self, image_path, output_path,
                num_passes=2, noise_std=8, blur_radius=1.2,
                resize_scale=0.92, jpeg_quality=75):
        """
        Applies iterative image processing to remove AI fingerprints.
        """
        import numpy as np
        from io import BytesIO

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        print(f"Processing image: {image_path} (size: {img.size})")

        processed = img.copy()

        for pass_num in range(num_passes):
            print(f"\n=== Pass {pass_num + 1}/{num_passes} ===")

            # Add Gaussian noise
            arr = np.array(processed).astype(float)
            noise = np.random.randn(*arr.shape) * noise_std
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            processed = Image.fromarray(arr)

            # Gaussian blur
            processed = processed.filter(
                ImageFilter.GaussianBlur(radius=blur_radius))

            # Resize down and back up
            w, h = processed.size
            new_w = int(w * resize_scale)
            new_h = int(h * resize_scale)
            processed = processed.resize((new_w, new_h), Image.LANCZOS)
            processed = processed.resize((w, h), Image.LANCZOS)

            # JPEG compression pass
            buf = BytesIO()
            processed.save(buf, format='JPEG', quality=jpeg_quality)
            buf.seek(0)
            processed = Image.open(buf).copy()

            print(f"  Applied: noise(std={noise_std}), blur(r={blur_radius}), "
                  f"resize({resize_scale}x), JPEG(q={jpeg_quality})")

        # Save final result
        processed.save(output_path, quality=95)
        print(f"\nConversion complete.")
        return output_path


def image_conversion(image_path, output_filename,
                     num_passes=2, noise_std=8, blur_radius=1.2,
                     resize_scale=0.92, jpeg_quality=75):
    """
    Converts an AI-generated image to bypass AI detectors using
    image processing techniques.

    Args:
        image_path: Path to the input AI-generated image
        output_filename: Path to save the converted image
        num_passes: Number of processing passes
        noise_std: Standard deviation of Gaussian noise
        blur_radius: Radius for Gaussian blur
        resize_scale: Scale factor for resize (down then up)
        jpeg_quality: JPEG quality for internal compression pass

    Returns:
        Path to the output image
    """
    converter = AdversarialImageConverter()
    result = converter.convert(
        image_path, output_filename,
        num_passes=num_passes,
        noise_std=noise_std,
        blur_radius=blur_radius,
        resize_scale=resize_scale,
        jpeg_quality=jpeg_quality
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

    def convert(self, text, max_retries=2):
        print(f"\nOriginal Text Start: {text[:80]}...")

        current_prob = self.get_probability(text, target_label_idx=0)
        print(f"Initial 'Real' Probability: {current_prob:.4f}")

        best_text = text
        best_prob = current_prob

        for outer_pass in range(max_retries):
            num_candidates = 8 + (outer_pass * 4)
            print(f"\n=== Pass {outer_pass + 1}/{max_retries} "
                  f"(candidates: {num_candidates}) ===")

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

                different_candidates = [c for c in candidates
                                        if c.strip() != original_sent.strip()]
                if not different_candidates:
                    print("    -> No different candidates generated.")
                    continue

                scored_candidates = []
                for candidate in different_candidates:
                    temp_sentences = sentences[:]
                    temp_sentences[i] = candidate
                    temp_full_text = " ".join(temp_sentences)
                    prob = self.get_probability(temp_full_text,
                                                target_label_idx=0)
                    scored_candidates.append((candidate, prob, temp_full_text))

                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                best_cand, best_cand_prob, best_cand_text = scored_candidates[0]

                sentences[i] = best_cand
                best_text = best_cand_text
                best_prob = best_cand_prob
                print(f"    -> Rewritten! Score: {best_prob:.4f}")

            print(f"\n  After pass {outer_pass + 1}: Score = {best_prob:.4f}")

        print(f"Conversion complete (score: {best_prob:.4f}).")
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
