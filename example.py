from medreviewanalyzer import MedicalReviewAnalyzer
from medreviewanalyzer.utils.metrics import compute_metrics


analyzer = MedicalReviewAnalyzer(verbose=True)

# Анализ одного отзыва
result = analyzer.analyze_review("отличное место , специалисты хорошие и внимательные , все объясняют , отвечают на все вопросы , персонал отзывчивый и всегда готов помочь , конечно ожидание бывает долгими , даже если по записи , но думаю стоит того , чтобы подождать")
print(result)

# # Анализ множества отзывов
import pandas as pd
df = pd.read_excel('example.xlsx')
result = analyzer.analyze_dataframe(df=df)

stats = compute_metrics([a for a in result['аспекты']])
analyzer.save_analysis_results(result, stats)