{
    dataset_reader: {
        type: 'uds_reader',
        lazy: false
    },
    train_data_path: 'data/train_instrument.json',
    validation_data_path: 'data/dev_instrument.json',
    model: {
        type: 'neural_davidsonian',
        embedder: {
            token_embedders: {
                tokens: {
                    type: 'embedding',
                    pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
                    embedding_dim: 50,
                    trainable: false
                }
            }
        },
        encoder: {
            type: 'lstm',
            input_size: 50,
            hidden_size: 25,
            bidirectional: true
        }
    },
    data_loader: {
        batch_size: 10,
        shuffle: true
    },
    trainer: {
        num_epochs: 10,
        patience: 3,
        cuda_device: -1,
        grad_clipping: 5.0,
        validation_metric: '-loss',
        optimizer: {
            type: 'adam',
            lr: 0.001
        }
    }
}