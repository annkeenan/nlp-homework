Baseline
--------

To test the accuracy of the unigram on the development set when trained on the
training data run the command:

python3 accuracy.py --model unigram --train english/train --predict english/dev

English
-------

To test the 5-gram on the development/test data:

python3 accuracy.py --model fivegram --train english/train --predict english/dev
python3 accuracy.py --model fivegram --train english/train --predict english/test

Testing the improved 7-gram:

python3 accuracy.py --train english/train --predict english/dev
python3 accuracy.py --train english/train --predict english/test

Chinese
-------

Testing the Chinese language model:

python3 chinese-accuracy.py --charmap chinese/charmap --train chinese/train.han --predict chinese/dev.pin --correct chinese/dev.han
python3 chinese-accuracy.py --charmap chinese/charmap --train chinese/train.han --predict chinese/test.pin --correct chinese/test.han
