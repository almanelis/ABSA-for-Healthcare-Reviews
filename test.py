# from pyabsa import AspectTermExtraction as ATEPC

# # Load a pre-trained aspect extractor
# aspect_extractor = ATEPC.AspectExtractor('multilingual')
# # Extract aspect terms from a single sentence
# aspect_extractor.predict(
#     'Дозвонилась на горячую линию для уточнения процедуры прикрепления к женской консультации с более удобной локацией, в связи с тем, что мою жк перенесли к черту на кулички.',
# )

from pyabsa import TextClassification as TC

classifier = TC.TextClassifier(checkpoint='multilingual')

# Пример текста
text = "Дозвонилась на горячую линию для уточнения процедуры прикрепления к женской консультации с более удобной локацией, в связи с тем, что мою жк перенесли к черту на кулички."

# Предсказание
result = classifier.predict(text)