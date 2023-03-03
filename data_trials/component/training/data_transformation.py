import os

from data_trials.entity.schema import NewsDataSchema
import sys
import re
import spacy
from spacy.lang.en import English
import numpy as np
from pyspark.sql.functions import udf,struct,lit,row_number
from pyspark.sql.window import Window
import contractions as cs
from deep_translator import GoogleTranslator
from flair.models import TextClassifier
from flair.data import Sentence
from pyspark.sql.functions import lower
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,when,regexp_replace
import pyspark.sql.functions as function
from pyspark.sql import Row


import string
import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import StandardScaler, VectorAssembler, OneHotEncoder, StringIndexer, Imputer
from pyspark.ml.pipeline import Pipeline
import pyarrow as pa
import pyarrow.parquet as pq

from data_trials.config.spark_manager import spark_session
from data_trials.exception import DataTrialException
from data_trials.logger import logger
from data_trials.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from data_trials.entity.config_entity import DataTransformationConfig
from pyspark.sql import DataFrame
from pandas import DataFrame as pandas_dataframe
from data_trials.ml.feature import FrequencyImputer, DerivedFeatureGenerator
from pyspark.ml.feature import IDF, Tokenizer, HashingTF
from pyspark.sql.functions import col, rand

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import regularizers
from numpy import asarray
from numpy import zeros


class DataTransformation():

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig,
                 schema=NewsDataSchema()
                 ):
        try:
            super().__init__()
            self.data_val_artifact = data_validation_artifact
            self.data_tf_config = data_transformation_config
            self.schema = schema
        except Exception as e:
            raise DataTrialException(e, sys)

    def read_data(self) -> pandas_dataframe:
        try:
            file_path = self.data_val_artifact.accepted_file_path
            dataframe: pandas_dataframe = pd.read_parquet(file_path)
            print(dataframe.info())
            return dataframe
        except Exception as e:
            raise DataTrialException(e, sys)
        
    def expand_contractions(self,dataframe : pandas_dataframe) -> pandas_dataframe:
        try:    
           for row in dataframe.iterrows():
                 text = row[1] 
                 print(text)              
                 text = ' '.join([cs.fix(word) for word in str(text).split()])  
                 print(text) 
                 row["lead_paragraph"] = text
           return dataframe
        except Exception as e:
            raise DataTrialException(e, sys)
    
    def ascii_ignore(self,dataframe : pandas_dataframe) -> pandas_dataframe:
      try:
        for row in dataframe.iterrows():
            text = row[1]            
            str(text).encode('ascii', 'ignore').decode('ascii')
            row[1] = text
        return dataframe
      except Exception as e:
            raise DataTrialException(e, sys)
        
    def create_Word_Embedding(self,vocab: int,tokenizer : Tokenizer) ->np.ndarray:
       try:
           embedding_dictionary = dict()
           glove_file = open('D:\\MLProjectNeuron\\dataTrials\\trial\\glove.6B.100d.txt',encoding='utf-8')
           for line in glove_file:
                 records = line.split()
                 word = records[0]
                 vector_dimensions = asarray(records[1:],dtype = 'float32')
                 embedding_dictionary[word] = vector_dimensions
           glove_file.close()
           
           
           embedding_matrix = zeros((vocab,100))
           for word,index in tokenizer.word_index.items():
               embedding_vector = embedding_dictionary.get(word)
               if embedding_vector is not None:
                        embedding_matrix[index] = embedding_vector
           return np.array(embedding_matrix)
       except Exception as e:
            raise DataTrialException(e, sys)        
    
         
    def remove_extra_spaces(self,dataframe : pandas_dataframe) -> pandas_dataframe:
       try:             
             for row in dataframe.iterrows():
                text = row[1] 
                re.sub(' +',' ',str(text))
                row[1] = text
             return dataframe
       except Exception as e:
             raise DataTrialException(e, sys)
       
    def stopwords_removal(self,dataframe : pandas_dataframe) -> pandas_dataframe:
       try: 
            
            lemmatizer = WordNetLemmatizer()
            corpus = []
            messages = dataframe['lead_paragraph'].tolist()            
            for i in range(0, len(messages)):
                review = messages[i]
                review = review.split()
                review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
                review = ' '.join(review)
                corpus.append(review)
            dataframe['lead_paragraph'] = pd.DataFrame(corpus)                                     
            return dataframe
       except Exception as e:
            raise DataTrialException(e, sys)
    
    def lemmatize_text(self,text:str) -> str:
        try:
              w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
              lemmatizer = WordNetLemmatizer()
              return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
        except Exception as e:
            raise DataTrialException(e, sys)
        
        
    def extract_target_column(self,dataframe : pandas_dataframe) -> pandas_dataframe:
       try:
                pd.set_option('display.max_colwidth', None)                
                new_df= dataframe.copy()
                new_df['sentence_lab']="Neutral"
                new_df['confidence']= 0.0
                #RUN FLAIR ALGO TO DETERMINE SENTIMENTS AND COPY THOSE IN SENTENCE_LAB column
                classifier = TextClassifier.load('en-sentiment') 
                              
                for index,row in new_df.iterrows(): 
                    text = row[0]                    
                    sentence = Sentence(text)
                    classifier.predict(sentence)
                    new_df.at[index, 'sentence_lab'] = sentence.labels[0].to_dict()['value']
                    new_df.at[index, 'confidence'] = sentence.labels[0].to_dict()['confidence']
                  
                
                
                new_df['sentence_lab'] = new_df['sentence_lab'].str.replace('POSITIVE','1')
                new_df['sentence_lab'] = new_df['sentence_lab'].str.replace('NEGATIVE','0')
                
                print("target labels created")
                
                return new_df
       except Exception as e:
                 raise DataTrialException(e, sys)
             
    def preprocessing_data(self,dataframe : pandas_dataframe) -> pandas_dataframe:        
            #PySpark DataFrame doesn’t contain the apply() function however,we can leverage Pandas DataFrame.apply() by running Pandas API over PySpark          
               #Use self.expand_contractions because Python doesn't understand that you want to access that method from the class,
#              # since you did not specify the class name as part of the method invocation. 
#              # As such, it is looking for the function in the global namespace and failed to find it.
   
#  #             In python, every method within a class takes self at the first argument. 
#  #          So simply replace def removePunc(myWord) with def removePunc(self, myWord) and continue that for all of the methods within the class.
   
            #Converting spark dataframe to Pandas dataframe
            #pandasDataFrame = dataframe.toPandas()            
            #pandasDataFrame.head(10)
        try:                     
            
            print(dataframe.info())
            print(dataframe.shape)
            index_list = list(range(0, 31000)) #limit to 30 records            
            dataframe.drop(dataframe.index[index_list], inplace =True)
            
            #drop index column
            dataframe = dataframe.reset_index(drop=True)
            print("New Info : ",dataframe.info())
            print("New Shape : ",dataframe.shape)
         #   w = Window().orderBy(lit('A'))
        #    dataframe = dataframe.withColumn("row_num", row_number().over(w))
          #  print("Row number column added")
         
            #Drop empty string rows from dataframe
            dataframe['lead_paragraph'].replace('', np.nan, inplace=True)
            dataframe['lead_paragraph']=dataframe['lead_paragraph'].apply(str)
            
            #logger.info(f"Number of row and columns: ", dataframe.shape())
            
           
            #dataframe = dataframe.withColumn('lead_paragraph', lower(dataframe.lead_paragraph))         
            #REmove digits and words containing numbers            
            dataframe['lead_paragraph']=dataframe['lead_paragraph'].apply(lambda x: re.sub(r'\w*\d\w*','', str(x)))
            print(dataframe.head(5))
            #Expand contractions words like we're ,they're etc. not working properly
         
            #Lower Case for all sentences in lead_paragraph column
            dataframe['lead_paragraph'] = dataframe['lead_paragraph'].apply(lambda x: str(x).lower())
               
            #dataframe = self.expand_contractions(dataframe)
            dataframe['lead_paragraph']=dataframe['lead_paragraph'].apply(lambda x:' '.join([cs.fix(word) for word in str(x).split()]))
            print("Contractions Expanded")    
        
           
            #Remove ASCII characters  
        
            dataframe['lead_paragraph']=dataframe['lead_paragraph'].apply(lambda x:str(x).encode('ascii', 'ignore').decode('ascii'))
               
                 
            #Remove Punctuation markers except hyphen
            remove = string.punctuation
            remove = remove.replace("-", "") # don't remove hyphens
            pattern = r"[{}]".format(remove) # create the pattern
            dataframe['lead_paragraph']=dataframe['lead_paragraph'].apply(lambda x: re.sub('[%s]' % re.escape(remove), '', str(x)))
                       
#              #Remove Hyphens with spaces
            #dataframe = self.remove_hyphens(dataframe)                  
            remove = '-'
            pattern = r"[{}]".format(remove) # create the pattern
            #weird punctuations marks were not handled
            dataframe['lead_paragraph']=dataframe['lead_paragraph'].apply(lambda x: re.sub('[%s]' % re.escape(remove), ' ', str(x)))
           # logger.info(f"Number of row: [{dataframe.count()}] and column: [{len(dataframe.columns)}]")     
            print("hyphen removed")          
              
#              #Translate to English all sentences
            #dataframe = self.translate_english(dataframe) 
            translator = GoogleTranslator()         
          #  dataframe['lead_paragraph']=dataframe['lead_paragraph'].apply(lambda x: translator.translate(str(x),target='en'))
            print("Sentences Translated")
#             dataframe = dataframe.toDF(*[re.sub('[%s]' % re.escape(remove), '', c) for c in dataframe.columns])   

             #5. Removing extra spaces
            dataframe['lead_paragraph']=dataframe['lead_paragraph'].apply(lambda x: re.sub(' +',' ',str(x)))   
            print("Extra spaces removed")
            
            dataframe = self.extract_target_column(dataframe)
           

            #7. Remove stopwords
            dataframe = self.stopwords_removal(dataframe)          
            print("Stopwords removed successfully")
            print(dataframe.head(5)) 
            
             #6. Lemmatization
            #dataframe = self.lemmatize_text(dataframe)
            dataframe['lead_paragraph']=dataframe['lead_paragraph'].apply(str)
            dataframe['lead_paragraph'] = dataframe.lead_paragraph.apply(lambda x: self.lemmatize_text(str(x)))
            print("Words Lemmatized")
                   
        
            #Remove punctation characters
            remove = string.punctuation                        
            dataframe['lead_paragraph']=dataframe['lead_paragraph'].apply(lambda x: re.sub('[%s]' % re.escape(remove), '', str(x)))
            print(dataframe.head(5))
            
            print("HEREEREEEEEE")      
            dataframe['sentence_lab'].astype(str).astype(int)
            print(dataframe.info())
            print(dataframe.head(5))
            dataframe.to_csv('D://content//newYork2019.csv')
            
            return dataframe
        except Exception as e:
            raise DataTrialException(e, sys)

    
    
    def get_data_transformation_pipeline(self, ) -> Pipeline:
        try:

            stages = [

            ]

            # numerical column transformation

            # generating additional columns
            derived_feature = DerivedFeatureGenerator(inputCols=self.schema.derived_input_features,
                                                      outputCols=self.schema.derived_output_features)
            stages.append(derived_feature)
            # creating imputer to fill null values
            imputer = Imputer(inputCols=self.schema.numerical_columns,
                              outputCols=self.schema.im_numerical_columns)
            stages.append(imputer)

            frequency_imputer = FrequencyImputer(inputCols=self.schema.one_hot_encoding_features,
                                                 outputCols=self.schema.im_one_hot_encoding_features)
            stages.append(frequency_imputer)
            for im_one_hot_feature, string_indexer_col in zip(self.schema.im_one_hot_encoding_features,
                                                              self.schema.string_indexer_one_hot_features):
                string_indexer = StringIndexer(inputCol=im_one_hot_feature, outputCol=string_indexer_col)
                stages.append(string_indexer)

            one_hot_encoder = OneHotEncoder(inputCols=self.schema.string_indexer_one_hot_features,
                                            outputCols=self.schema.tf_one_hot_encoding_features)

            stages.append(one_hot_encoder)

            tokenizer = Tokenizer(inputCol=self.schema.tfidf_features[0], outputCol="words")
            stages.append(tokenizer)

            hashing_tf = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=40)
            stages.append(hashing_tf)
            idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol=self.schema.tf_tfidf_features[0])
            stages.append(idf)

            vector_assembler = VectorAssembler(inputCols=self.schema.input_features,
                                               outputCol=self.schema.vector_assembler_output)

            stages.append(vector_assembler)

            standard_scaler = StandardScaler(inputCol=self.schema.vector_assembler_output,
                                             outputCol=self.schema.scaled_vector_input_features)
            stages.append(standard_scaler)
            pipeline = Pipeline(
                stages=stages
            )
            logger.info(f"Data transformation pipeline: [{pipeline}]")
            print(pipeline.stages)
            return pipeline

        except Exception as e:
            raise DataTrialException(e, sys)

    def get_balanced_shuffled_dataframe(self, dataframe: DataFrame) -> DataFrame:
        try:

            count_of_each_cat = dataframe.groupby(self.schema.target_column).count().collect()
            label = []
            n_record = []
            for info in count_of_each_cat:
                n_record.append(info['count'])
                label.append(info[self.schema.target_column])

            minority_row = min(n_record)
            n_per = [minority_row / record for record in n_record]

            selected_row = []
            for label, per in zip(label, n_per):
                print(label, per)
                temp_df, _ = dataframe.filter(col(self.schema.target_column) == label).randomSplit([per, 1 - per])
                selected_row.append(temp_df)

            selected_df: DataFrame = None
            for df in selected_row:
                df.groupby(self.schema.target_column).count().show()
                if selected_df is None:
                    selected_df = df
                else:
                    selected_df = selected_df.union(df)

            selected_df = selected_df.orderBy(rand())

            selected_df.groupby(self.schema.target_column).count().show()
            return selected_df
        except Exception as e:
            raise DataTrialException(e, sys)

    def update_modeltrainerconfig(self,modeltrainer_config_file : str,vocab_length : int ,maxlen : int) :
        try:
       
            with open(modeltrainer_config_file, "r") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                # if dictionary is found
                if "VOCAB_LEN" in line:
                    # replace the value of this key
                        value = line.split("=")[1]                    
                        lines[i] = line.replace(value, str(vocab_length)+"\n")                                                
                if "MAX_LEN" in line:
                    # replace the value of this key
                        value = line.split("=")[1]                    
                        lines[i] = line.replace(value, str(maxlen))
                        break
        
            with open(modeltrainer_config_file, "w") as f1:
                f1.writelines(lines)
        except Exception as e:
            raise DataTrialException(e, sys)
        finally : 
            f.close()
            f1.close()

            
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info(f">>>>>>>>>>>Started data transformation <<<<<<<<<<<<<<<")
            #https://www.kdnuggets.com/2020/03/tensorflow-keras-tokenization-text-data-prep.html
            #dataframe: DataFrame = self.read_data()
            dataframe: pandas_dataframe = self.read_data()
            # dataframe = self.get_balanced_shuffled_dataframe(dataframe=dataframe)
            logger.info(f"Number of row: [{dataframe.count()}] and column: [{len(dataframe.columns)}]")
            
            pd.set_option('display.max_colwidth', None)                
            dataframe = self.preprocessing_data(dataframe)
            
            X = dataframe['lead_paragraph']
            y = dataframe['sentence_lab'].astype(str).astype(int)
            
            y = np.array(list(map(lambda x: 1 if x==1 else 0,y)))
            
            test_size = self.data_tf_config.test_size
            logger.info(f"Splitting dataset into train and test set using ration: {1 - test_size}:{test_size}")
            #train_dataframe, test_dataframe = dataframe.randomSplit([1 - test_size, test_size])
            train_dataframe, test_dataframe, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
            #logger.info(f"Train dataset has number of row: [{train_dataframe.shape}] and"
             #            f" column: [{train_dataframe.count()}]")
            #logger.info(f"Train dataset has number of row: [{test_dataframe.shape}] and"
             #            f" column: [{test_dataframe.count()}]")

            #Embedding layer expects words to be in numeric form  # Using Tokenizer function from keras.preprocessing.text library
            #Method fit_on_texts trains the tokenizer #Method texts_to_sequence converts sentences to their numeric form
            tokenizer = Tokenizer()
            
            #Tokenizer has been fit on Texts of X Train only and not X Test
            tokenizer.fit_on_texts(train_dataframe)


            #Tokenizer.fit_to_texts(dictionary) does the word indexing, i.e. it builds a translation of your any sequence of words to numbers(vector representation),
            # so it might be that vocabulary difference between the training and test set is not a null set, i.e. some of the words in test are not present in the word indexer built by the Tokenizer object if it used only the train data. 
            # Which could result in some test set generating different vector if you'd have trained your tokenizer only on training set
            X_train = tokenizer.texts_to_sequences(train_dataframe)
            X_test = tokenizer.texts_to_sequences(test_dataframe)

            #Now let's use our tokenizer to tokenize the test data, and then similarly encode our sequences. 
            # This is all quite similar to the above. Note that we are using the same tokenizer we created for training in order to facilitate simpatico between the 2 datasets, using the same vocabulary. 
            # We also pad to the same length and specifications as the training sequences.
            
            vocab_length = len(tokenizer.word_index)+1
            print(vocab_length)
            
            # logger.info(f"Train dataset has number of row: [{train_dataframe.count()}] and"
            #             f" column: [{len(train_dataframe.columns)}]")

            #Padding all reviews to fixed length  #words with less than 150 ,we add zeroes to make them 150
            maxlen = max([len(x) for x in X_train])       
            X_train = pad_sequences(X_train,padding = 'post',maxlen = maxlen)
            X_test = pad_sequences(X_test,padding = 'post',maxlen = maxlen)
            
            embedding = self.create_Word_Embedding(vocab_length,tokenizer)
            print(embedding.shape)

            
            # transformed_trained_dataframe = transformed_pipeline.transform(train_dataframe)
            # transformed_trained_dataframe = transformed_trained_dataframe.select(required_columns)

            # transformed_test_dataframe = transformed_pipeline.transform(test_dataframe)
            # transformed_test_dataframe = transformed_test_dataframe.select(required_columns)

            export_pipeline_file_path = self.data_tf_config.export_pipeline_dir

             # creating required directory
            os.makedirs(export_pipeline_file_path, exist_ok=True)
            os.makedirs(self.data_tf_config.transformed_test_dir, exist_ok=True)
            os.makedirs(self.data_tf_config.transformed_train_dir, exist_ok=True)
            transformed_train_data_file_path = os.path.join(self.data_tf_config.transformed_train_dir,
                                                            self.data_tf_config.file_name
                                                             )
            transformed_test_data_file_path = os.path.join(self.data_tf_config.transformed_test_dir,
                                                            self.data_tf_config.file_name
                                                            )
            transformed_train_data_file_path = transformed_train_data_file_path +'.txt'
            transformed_test_data_file_path = transformed_test_data_file_path +'.txt'
            # logger.info(f"Saving transformation pipeline at: [{export_pipeline_file_path}]")
            # transformed_pipeline.save(export_pipeline_file_path)
            # logger.info(f"Saving transformed train data at: [{transformed_train_data_file_path}]")
            # print(transformed_trained_dataframe.count(), len(transformed_trained_dataframe.columns))
            #pa_table = pa.table({"X": embedding[:, 0]}) #Failing here [only handle 1-dimensional arrays] ] ] ] ]
           # pq.write_table(pa_table, transformed_train_data_file_path +".parquet")
            print(embedding)
            print(X_test)
            
            np.savetxt(transformed_train_data_file_path, embedding,encoding='utf-8')
            np.savetxt(transformed_test_data_file_path, X_test,encoding='utf-8')
            
            #Test data should only and only be used for testing. You can consider them as unseen data. So,you should not train your word-embedding with test data.

            # logger.info(f"Saving transformed test data at: [{transformed_test_data_file_path}]")
            # print(transformed_test_dataframe.count(), len(transformed_trained_dataframe.columns))
            #pa1_table = pa.table({"testData": X_test[:,0]})
          #  pq.write_table(pa1_table, transformed_test_data_file_path +".parquet")

            data_tf_artifact = DataTransformationArtifact(
                 transformed_train_file_path=transformed_train_data_file_path,
                 transformed_test_file_path=transformed_test_data_file_path,
                 exported_pipeline_file_path=export_pipeline_file_path,

             )

            
            modeltrainer_config_file = "D:\\MLProjectNeuron\\dataTrials\\data_trials\\constant\\training_pipeline_config\\model_trainer.py"
            self.update_modeltrainerconfig(modeltrainer_config_file,vocab_length,maxlen)
          
            logger.info(f"Data transformation artifact: [{data_tf_artifact}]")
            return data_tf_artifact
        except Exception as e:
            raise DataTrialException(e, sys)
        
