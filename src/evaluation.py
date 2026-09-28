"""
RAGAS evaluation system for RAG pipeline
"""
from typing import List, Dict, Any, Optional
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from src.models import EvaluationResponse


class RAGASEvaluator:
    """Evaluate RAG pipeline using RAGAS metrics"""
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    
    def evaluate(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> EvaluationResponse:
        """
        Evaluate RAG pipeline on a set of questions
        
        Args:
            questions: List of questions to evaluate
            ground_truths: Optional list of ground truth answers for context_recall
        
        Returns:
            EvaluationResponse with RAGAS metrics
        """
        if not questions:
            raise ValueError("Questions list cannot be empty")
        
        print(f"Evaluating {len(questions)} questions...")
        
        # Generate answers and collect data
        eval_data = []
        details = []
        
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}...")
            
            try:
                # Get response from RAG pipeline
                response = self.rag_pipeline.query(question)
                
                # Collect contexts from retrieved documents
                contexts = [source.excerpt for source in response.sources]
                
                # Prepare data point
                data_point = {
                    "question": question,
                    "answer": response.answer,
                    "contexts": contexts,
                }
                
                # Add ground truth if available
                if ground_truths and i < len(ground_truths):
                    data_point["ground_truth"] = ground_truths[i]
                
                eval_data.append(data_point)
                
                # Store details
                details.append({
                    "question": question,
                    "answer": response.answer,
                    "confidence": response.confidence,
                    "num_sources": len(response.sources),
                    "retrieval_time_ms": response.retrieval_time_ms,
                    "generation_time_ms": response.generation_time_ms
                })
            
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                details.append({
                    "question": question,
                    "error": str(e)
                })
        
        if not eval_data:
            raise ValueError("No valid evaluation data generated")
        
        # Create dataset
        dataset = Dataset.from_list(eval_data)
        
        # Select metrics based on available data
        metrics_to_use = [faithfulness, answer_relevancy, context_precision]
        
        # Only include context_recall if ground truths are provided
        if ground_truths and len(ground_truths) > 0:
            metrics_to_use.append(context_recall)
        
        # Run RAGAS evaluation
        print("Running RAGAS evaluation...")
        try:
            results = evaluate(dataset, metrics=metrics_to_use)
            
            # Extract scores
            faithfulness_score = results.get('faithfulness', 0.0)
            context_precision_score = results.get('context_precision', 0.0)
            answer_relevance_score = results.get('answer_relevancy', 0.0)
            context_recall_score = results.get('context_recall', None) if ground_truths else None
            
            # Calculate overall score
            scores = [faithfulness_score, context_precision_score, answer_relevance_score]
            if context_recall_score is not None:
                scores.append(context_recall_score)
            
            overall_score = sum(scores) / len(scores)
            
            print(f"✅ Evaluation complete!")
            print(f"   Faithfulness: {faithfulness_score:.3f}")
            print(f"   Context Precision: {context_precision_score:.3f}")
            print(f"   Answer Relevance: {answer_relevance_score:.3f}")
            if context_recall_score is not None:
                print(f"   Context Recall: {context_recall_score:.3f}")
            print(f"   Overall Score: {overall_score:.3f}")
            
            return EvaluationResponse(
                faithfulness=faithfulness_score,
                context_precision=context_precision_score,
                answer_relevance=answer_relevance_score,
                context_recall=context_recall_score,
                overall_score=overall_score,
                details=details
            )
        
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            # Return fallback response with basic metrics
            return EvaluationResponse(
                faithfulness=0.0,
                context_precision=0.0,
                answer_relevance=0.0,
                context_recall=None,
                overall_score=0.0,
                details=details
            )


def create_evaluation_dataset(
    questions: List[str],
    answers: Optional[List[str]] = None,
    contexts: Optional[List[List[str]]] = None,
    ground_truths: Optional[List[str]] = None
) -> Dataset:
    """
    Create a RAGAS evaluation dataset
    
    Args:
        questions: List of questions
        answers: Optional list of generated answers
        contexts: Optional list of retrieved contexts for each question
        ground_truths: Optional list of ground truth answers
    
    Returns:
        Dataset object for RAGAS evaluation
    """
    data = []
    
    for i, question in enumerate(questions):
        data_point = {"question": question}
        
        if answers and i < len(answers):
            data_point["answer"] = answers[i]
        
        if contexts and i < len(contexts):
            data_point["contexts"] = contexts[i]
        
        if ground_truths and i < len(ground_truths):
            data_point["ground_truth"] = ground_truths[i]
        
        data.append(data_point)
    
    return Dataset.from_list(data)
