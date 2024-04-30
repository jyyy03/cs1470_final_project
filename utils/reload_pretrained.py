import tensorflow as tf

def restore_model_from_checkpoint(checkpoint_path, model):
    reader = tf.train.load_checkpoint(checkpoint_path)
    restore_dict = reader.get_variable_to_shape_map()
    for var in model.variables:
       if var.shape in restore_dict:
            tensor = reader.get_tensor(var.path)
            var.assign(tensor)
            print(f'Successfully loaded {var.path} into the model.')
            
    print("Model restored with subset matching from:", checkpoint_path)