# gpt2_model.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import torch
import os

class TextDataset(Dataset):
    """
    Dataset personalizado para cargar los datos de texto.
    """
    def __init__(self, tokenizer, file_path, block_size=128):
        self.examples = []

        # Verifica si el archivo existe
        if not os.path.isfile(file_path):
            raise ValueError(f"El archivo {file_path} no se encuentra.")

        # Cargar y leer el archivo de texto
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        # Tokenizar el texto y dividirlo en bloques
        tokenized_text = tokenizer.encode(text)
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(tokenized_text[i:i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class GPT2Model:
    def __init__(self, model_name="gpt2"):
        """
        Inicializa el modelo GPT-2 y el tokenizador.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Usa GPU si está disponible
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def load_dataset(self, file_path, block_size=128):
        """
        Carga el dataset desde el archivo de texto preprocesado.
        """
        # Aquí se cargará el archivo company_names.txt y se convertirá en un dataset tokenizado
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=file_path,
            block_size=block_size  # Ajusta el block_size según el tamaño de los nombres
        )
        return dataset

    def fine_tune(self, dataset_path, output_dir='./fine_tuned_gpt2'):
        """
        Ajuste fino del modelo GPT-2 usando el archivo de texto preprocesado.
        """
        # Verificar que el archivo company_names.txt existe y no está vacío
        if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) == 0:
            raise ValueError(f"El archivo {dataset_path} no existe o está vacío.")

        # Cargar el dataset desde el archivo
        dataset = self.load_dataset(dataset_path)

        # Verificar si el dataset tiene ejemplos válidos
        if dataset is None or len(dataset) == 0:
            raise ValueError(f"El dataset está vacío o no tiene ejemplos válidos: {dataset_path}")

        # Configurar los argumentos del entrenamiento
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,  # Número de épocas de entrenamiento
            per_device_train_batch_size=4,  # Tamaño del batch
            save_steps=10_000,  # Guardar cada 10,000 pasos
            save_total_limit=2,  # Guardar solo los 2 últimos checkpoints
        )

        # Configurar la forma en la que el dataset se pasa al modelo
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False  # No enmascarar las palabras como en BERT
        )

        # Crear el entrenador del modelo
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset  # Dataset para entrenar
        )

        # Entrenar el modelo
        trainer.train()

        # Guardar el modelo ajustado
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Modelo guardado en {output_dir}.")

    def generate_name(self, input_text, max_length=20, num_return_sequences=5, temperature=1.0):
        """
        Genera nombres de empresas a partir del modelo GPT-2 entrenado.
        """
        # Codificar el texto de entrada
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        # Generar nombres usando sampling
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=True  # Sampling para generar múltiples secuencias
        )

        # Decodificar las secuencias generadas a texto
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
