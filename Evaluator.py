from EvaluationData import CreateDataSets
from EvaluatedAlgorithm import EvaluatedAlgorithm

class Evaluator():
    
    def __init__(self, dataset, rankings, diversity=False, leave_one_out=False, anti_test=False):
        self.models = []
        ed = CreateDataSets(dataset, rankings, diversity=diversity, leave_one_out=leave_one_out, anti_test=anti_test)
        self.dataset = ed
    
    def add_model(self, model, name):
        m = EvaluatedAlgorithm(model, name)
        self.models.append(m)
        
    def evaluate(self, topN):
        results = {}
        for model in self.models:
            print("Evaluating ", model.name, "...")
            results[model.name] = model.evaluate_model(self.dataset, topN)

        # Print results
        print("\n")
        
        if (topN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                                      metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))

        
    def recommend_top_books(self, book_ds, test_user_id=85, k=10):
        
        for model in self.models:
            print("\nUsing recommender ", model.name)
            
            print("\nBuilding recommendation model...")
            train = self.dataset.get_full_train()
            model.model.fit(train)
            
            print("Computing recommendations...")
            test = self.dataset.get_anti_test_for_user(test_user_id)
        
            predictions = model.model.test(test)
            
            recommendations = []
            
            print ("\nWe recommend:")
            for user_id, book_id, y_true, y_pred, _ in predictions:
                recommendations.append((book_id, y_pred))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for ratings in recommendations[:k]:
                print(book_ds.get_book_name(ratings[0]), round(ratings[1], 2))   