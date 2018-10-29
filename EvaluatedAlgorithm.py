from RecommenderMetrics import RecommenderMetrics
from EvaluationData import CreateDataSets
from datetime import datetime

class EvaluatedAlgorithm:
    
    def __init__(self, model, name):
        self.model = model
        self.name = name
        
    def evaluate_model(self, evaluationData, topN, n=10, verbose=True):
        metrics = {}
        # Compute accuracy
        if (verbose):
            print("Evaluating accuracy...")
        t0 = datetime.now()
        self.model.fit(evaluationData.get_train())
        print("Training Time: {}".format(datetime.now() - t0))
        t1 = datetime.now()
        predictions = self.model.test(evaluationData.get_test())
        print("Prediction Time: {}".format(datetime.now() - t1))
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)
        
        if (topN):
            # Evaluate top-10 with Leave One Out testing
            if (verbose):
                print("Evaluating top-N with leave-one-out...")
            self.model.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self.model.test(evaluationData.get_LOOCV_test())        
            # Build predictions for all ratings not in the training set
            allPredictions = self.model.test(evaluationData.get_LOOCV_anti_test())
            # Compute top 10 recs for each user
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions)   
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions)
            # Compute ARHR
            metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)
        
            #Evaluate properties of recommendations on full training set
            if (verbose):
                print("Computing recommendations with full data set...")
            self.algorithm.fit(evaluationData.get_full_train())
            allPredictions = self.model.test(evaluationData.get_full_anti_test())
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 8.0:
            metrics["Coverage"] = RecommenderMetrics.UserCoverage(  topNPredicted, 
                                                                   evaluationData.get_full_train().n_users, 
                                                                   ratingThreshold=8.0)
            # Measure diversity of recommendations:
            metrics["Diversity"] = RecommenderMetrics.Diversity(topNPredicted, evaluationData.get_similarities())
            
            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = RecommenderMetrics.Novelty(topNPredicted, 
                                                            evaluationData.get_popularity_rankings())
        
        if (verbose):
            print("Analysis complete.")
    
        return metrics