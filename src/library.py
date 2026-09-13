import cv2 as cv

class Card:
    """
    An object representing a card, containing its ID, keypoints, descriptors, and images.
    """
    def __init__(self, 
                card_id,
                keypoints = None,
                descriptors = None,
                image=None,
                normalized_image=None,
                ):
        
        self.card_id = card_id
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.image = image
        self.normalized_image = normalized_image

class CardLibrary:
    """
    A dictionary-based library for storing and managing Card objects.
    Each card is represented by a Card object, which contains its ID, keypoints, descriptors, and images.
    """
    def __init__(self, cards=None):
        self.cards = {} if cards is None else cards

    def add_card(self, card: Card):
        self.cards[card.card_id] = card

    @classmethod
    def from_directory(cls, directory, feature_extractor=None):
        instance = cls()

        """
        Load card images from a directory and optionally extract their features.

        Args:
            directory: The path to the directory containing card images.
            feature_extractor: A callable that takes an image and returns its features.

        Returns:
            self: The updated CardLibrary instance with loaded cards.
        """
        import os
        for filename in os.listdir(directory):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                card_id = os.path.splitext(filename)[0]
                image = cv.imread(os.path.join(directory, filename), cv.IMREAD_COLOR)
                normalized_image = cv.normalize(cv.cvtColor(image, cv.COLOR_BGR2GRAY), None, 0, 255, cv.NORM_MINMAX)
                # Testing just training on the text portion of the image:
                # image_text = normalized_image[:150, :280]
                # image_requirement = normalized_image[-150:]
                keypoints, descriptors = ([], None) if feature_extractor is None else feature_extractor(normalized_image)
                card = Card(card_id=card_id, keypoints=keypoints, descriptors=descriptors, image=image, normalized_image=normalized_image)
                instance.add_card(card)
        return instance

if __name__ == "__main__":
    library = CardLibrary.from_directory("data/saved_images/contracts")