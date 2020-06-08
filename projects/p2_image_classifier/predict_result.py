class Predict_Result:
    
    def __init__(self, probability, id, category_name):
        self.probability = probability
        self.id = id
        self.category_name = category_name
        
    def __str__(self):
        if self.category_name is None:
            return '(Probability: ' + str(self.probability) + ' , Label: ' + str(self.id) +')'
        else:
            return '(Probability: ' + str(self.probability) + ' , Category: ' + self.category_name +')'