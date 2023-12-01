### PRT

**Exploration 4 CIFAR-10 이미지 생성하기**

코더 : 최재혁
리뷰어 : 김연

- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**  
    * 문제에서 요구하는 최종 결과물이 첨부되었는지 확인  
    * 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 퀘스트 문제 요구조건 등을 지칭  
    * 해당 조건을 만족하는 부분의 코드 및 결과물을 근거로 첨부  

```
def train(dataset, epochs, save_every):
    start = time.time()
    history = {'gen_loss':[], 'disc_loss':[], 'real_accuracy':[], 'fake_accuracy':[]}

    for epoch in range(epochs):
        epoch_start = time.time()
        for it, image_batch in enumerate(dataset):
            gen_loss, disc_loss, real_accuracy, fake_accuracy = train_step(image_batch)
            history['gen_loss'].append(gen_loss)
            history['disc_loss'].append(disc_loss)
            history['real_accuracy'].append(real_accuracy)
            history['fake_accuracy'].append(fake_accuracy)

            if it % 50 == 0:
                display.clear_output(wait=True)
                generate_and_save_images(generator, epoch+1, it+1, noise_vector)
                print('Epoch {} | iter {}'.format(epoch+1, it+1))
                print('Time for epoch {} : {} sec'.format(epoch+1, int(time.time()-epoch_start)))

        if (epoch + 1) % save_every == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epochs, it, noise_vector)
        print('Time for training : {} sec'.format(int(time.time()-start)))

        draw_train_history(history, epoch)
```

>네, CIFAR-10 데이터셋을 활용하여 GAN 모델을 구현했고, 학습을 진행한 후 결과를 시각화로 표현된 결과를 확인할 수 있습니다.
    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    * 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    * 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    * 주석을 보고 코드 이해가 잘 되었는지 확인
    * 잘 작성되었다고 생각되는 부분을 근거로 첨부합니다.

```
# 인코더 구조
latent_dim = 64  # 잠재 공간의 차원

encoder_inputs = Input(shape=(32, 32, 3))
x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

encoder = Model(encoder_inputs, [z_mean, z_log_var], name='encoder')
encoder.summary()
```

>CIFAR_10.ipynb 파일의 대부분의 코드나 주석이 제가 작업한 코드 및 주석과 동일해서 이해가 어려운 부분은 없었습니다. Optional 파일의 VAE 모델 구조는 새로운 개념이라 새로웠고, 코드를 이해하는 것이 어려웠습니다. 특히 잠재 공간의 차원이 무엇을 의미하는지 궁금했습니다.

- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    * 문제 원인 및 해결 과정을 잘 기록하였는지 확인 또는
    * 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 실험이 기록되어 있는지 확인
    * 잘 작성되었다고 생각되는 부분을 근거로 첨부합니다.
        
>Optional 파일 참조. VAE GAN 모델 구조를 활용하여 이미지 데이터 생성을 실험하셨습니다.

- [X]  **4. 회고를 잘 작성했나요?**
    * 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해 배운점과 아쉬운점, 느낀점 등이 상세히 기록되어 있는지 확인
    * 딥러닝 모델의 경우, 인풋이 들어가 최종적으로 아웃풋이 나오기까지의 전체 흐름을 도식화하여 모델 아키텍쳐에 대한 이해를 돕고 있는지 확인

>실험 환경 구성이나 설정이 중요하다는 점과 연관하여 특히 이미지 데이터를 활용하는 경우 모델 학습 시 시간이 많이 소요되어 전체 과정에 영향을 주는 경험을 하셨습니다. 최근 이미지 데이터를 사용하는 실습이 많았어서 비슷한 고민이 있었고 각자의 해결 방안에 대해 공유할 수 있었습니다.

- [ ]  **5. 코드가 간결하고 효율적인가요?**
    * 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    * 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 모듈화(함수화) 했는지
    * 잘 작성되었다고 생각되는 부분을 근거로 첨부합니다.
    
>간결한 코드로 모델을 제작하였고, 학습 후 결과를 시각화하여 표현하였습니다. CIFAR-10 데이터셋의 이미지 데이터가 각각 GAN 모델에 의해, 즉 Optional 실험에서 추출한 결과 이미지 데이터의 경우 기존의 노드에서 학습한 GAN 모델에서 추출한 결과 이미지 데이터와는 다른 형태의 이미지 데이터로 추출되었습니다. 새롭고 흥미로운 결과라고 생각합니다.

---

# 회고 
우선 가장 크게 느끼는 것은 GPU 환경에 대한 선택이다.. 기존 프로젝트에서는 시간적으로 여유가 있어서 괜찮았지만 이번 GAN 모델을 활용하는 데 있어서 Colab 기본 gpu는 굉장히 많은 시간을 낭비하게 하였다.(1 EPOCH에 10분 이상.. ) 이에 새로이 GPU환경을 개선 하기 위해 Colab pro로 결제를 진행하여 다시 코드를 돌렸고 정말 높은 성능을 보여 시간적인 문제는 해결할수 있게 되었다. 또한 사진을 생성한다는 개념에 대해 직접 경험할수 있는 기회가 생겨서 너무 감사하고, 사실 아직도 만족하지 못하기에 더욱 좋은 성능의 결과를 내기 위해 다시 시도해 보려고 한다.
