"""
신윤중
2017-5-20
lab00 - install & setting
"""
# 텐서플로우 설치 및 세팅
# 아나콘다 활용

# 쉘을 통해 conda virtual env 생성
# conda create -n tensorflow35 python=3.5
""" 
3.5버전 파이썬을 기반으로 한 tensorflow35 이름의 virtual env 를 생성한다.
2017-5-20일 기준 tensorflow 는 python3.5.x 버전에서만 구동됨
"""

# activate virtual env
# activate tensorflow35
""" tensorflow35 이름을 가진 virtual env 활성화"""
# deactivate tensorflow35
""" tensorflow35 이름을 가진 virtual env 비활성화"""

# tensorflow package 설치
# pip install tensorflow

# package import 및 version 확인
import tensorflow as tf
tf.__version__

# jupyter notebook 에서 사용
# activate tensorflow35
# pip install nb_anaconda
# jupyter notebook
"""생성한 virtual env 에 nb_anaconda 를 설치해줘야 notebook 상에서 virtual env 를 사용할 수 있다."""
