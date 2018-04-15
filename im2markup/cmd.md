cmd_orders
==========
## data prepare
### First of all-file encoding
```
 In the first, you should see the encoding of the file 'im2latex_formulas.lst'. eapically process the labels
 1.Open this file in vim and in the 'esc' model
 2.Type the cmd ':set fileenoding' then you can see  the encoding of this file.
 3.Make this file encoding 'utf-8': type the cmd ': set fileencoding = utf-8'
 4.Check the change: again type the cmd:':set fileencoding' and see the file encoding.

```

### images
```
cd im2markup
python3 scripts/preprocessing/preprocess_images.py --input-dir /Users/xiaofeng/Code/Github/dataset/formula/generate/data_formula_ori --output-dir /Users/xiaofeng/Code/Github/dataset/formula/generate/images_processed
```

### labels

```
python3 scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file /Users/xiaofeng/Code/Github/im2latex-tensorflow/im2latex-dataset/generate/im2latex_formulas.lst --output-file /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst
```

### train.filter
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /Users/xiaofeng/Code/Github/dataset/formula/generate/images_processed --label-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst --data-path /Users/xiaofeng/Code/Github/im2latex-tensorflow/im2latex-dataset/generate/train.list --output-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/train_filter.lst
```

### validate.filter
```
python3 scripts/preprocessing/preprocess_filter.py --filter --image-dir /Users/xiaofeng/Code/Github/dataset/formula/generate/images_processed --label-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst --data-path /Users/xiaofeng/Code/Github/im2latex-tensorflow/im2latex-dataset/generate/validate.list --output-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/validate_filter.lst
```
### vocabulary
```
python3 scripts/preprocessing/generate_latex_vocab.py --data-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/train_filter.lst --label-path /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/formulas.norm.lst --output-file /Users/xiaofeng/Code/Github/dataset/formula/generate/prepared/latex_vocab.txt
```





