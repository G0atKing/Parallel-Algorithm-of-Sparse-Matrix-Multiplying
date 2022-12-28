#include<cstdlib>
#include<algorithm>
#include<iostream>
#include<ctime>
#include<pthread.h>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include<pmmintrin.h>
#include<xmmintrin.h>
#include <immintrin.h> //AVX、AVX2
using namespace std;
//生成向量
void generate_vector(int n,float* & x){
    x=new float[n];
    for(int i=0;i<n;i++)
        x[i]=rand()%10+1;
}

//生成稠密矩阵
void generate_matrix(float** & m,int n){
   m=new float*[n];
   for(int i=0;i<n;i++)
       m[i]=new float[n];
   for(int i=0;i<n;i++)
   {
       for(int j=0;j<n;j++)
      {
          int x=rand()%10;
          m[i][j]=x+1;
      }
   }
}

//生成稀疏矩阵
void generate_sparse_matrix(float** & m,int n,double s){
    //注：s为稀疏度
   m=new float*[n];
   for(int i=0;i<n;i++)
       m[i]=new float[n];
   for(int i=0;i<n;i++)
      for(int j=0;j<n;j++)
      {
          int x=rand()%1000;
          if(x>=1000*s)
            m[i][j]=0;//0位稀疏位置
          else
            m[i][j]=x%10+1;
      }
   return;
}

//一些变量进行定义：全局变量
const unsigned long Converter = 1000*1000 ; // 1s == 1000 * 1000 us


typedef struct{
	int	threadId;
} threadParm_t;

typedef struct{
	int	threadId;
	int rowid;
} threadParm_t2;

int n = 4096;//矩阵规模
int THREAD_NUM=0;//线程的个数
int nonzeros=0;//稀疏矩阵中非零元素的个数
int nozerorows=0;//稀疏矩阵中不全为0的行数，这个变量是未来进行稀疏矩阵pThread算法优化的关键变量
int single_circle=10;//单个线程的工作量
double s=0.005;

//一些openMP中用到的变量的值
int OMP_NUM_THREADS=16;

float **mat=NULL;//稀疏矩阵
float **mat_nonsparse=NULL;//稠密矩阵
float **mat_res1=NULL;//结果矩阵1
float **mat_res2=NULL;//结果矩阵2
float **mat_res3=NULL;//结果矩阵2
float *vec=NULL;//向量
float *y=NULL;//spmv结果1
float *yy=NULL;//spmv结果2
float *yyy=NULL;//spmv结果3


//稀疏矩阵表示法：在pThread编程中，为了方便起见，我们将所有行的首个元素的下标都存储在index数组中
float *value=NULL;
int *col=NULL;
int *row=NULL;
int *index=NULL;

//将稀疏矩阵转化为COO表示的格式
int matrix_to_coo(float **M,int n,float* &value,int* & row,int* & col,int* & index){
    //n为矩阵行列数 nonzeros代表矩阵的非零元素个数
   int i,j;
   int a=0;
   for(i=0;i<n;i++)
      for(j=0;j<n;j++)
          if(M[i][j]!=0)
              a++;//统计非零元素个数
   value=new float[a];
   col=new int[a];
   row=new int[a];
   int k=0;
   for(i=0;i<n;i++)
   {
      for(j=0;j<n;j++)
      {
          if(M[i][j]!=0)
          {
              row[k]=i;
              col[k]=j;
              value[k++]=M[i][j];//COO格式
          }
      }
   }

   for(int k=1;k<a;k++){
      if(row[k]!=row[k-1]){
          nozerorows++;
      }
   }
   nozerorows=nozerorows+1;
   index=new int[nozerorows+1];
   int p=0;
   index[p++]=0;
   for(int k=1;k<a;k++){
      if(row[k]!=row[k-1]){
          index[p++]=k;  //我们将所有行的首个元素的下标(COO下标)都存储在index数组中
      }
   }
   index[nozerorows]=nonzeros;//这里是一个哨兵
   return a;
}

//实现COO与稠密矩阵相乘串行算法
double coo_multiply_matrix_serial(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    struct timeval val,newVal; //时间
    gettimeofday(&val, NULL);//起始时间
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            c[i][j]=0;//结果矩阵初始化

    for (int i=0;i<nonzeros;i++)//所有非0元
        for(int k=0;k<n;k++)//列循环
            c[row[i]][k] += value[i] * b[col[i]][k];

    gettimeofday(&newVal, NULL);//结束时间
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//实现COO与稠密矩阵相乘并行算法SEE2
//基本思路是循环展开
double coo_multiply_matrix_sse(int nonzeros,int n,int* row,int* col,float* value,float**b,float**c){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    __m128 t1,t2,t3,sum;
    int choice = n % 4;
    for (int i=0;i<nonzeros;i++)
    {
        for(int k=0;k<n-choice;k+=4)//一次求4组
        {
            t1=_mm_load_ps(b[col[i]]+k);//t1=b[col[i]][k]
            sum = _mm_setzero_ps();//sum=0
            t3 = _mm_set_ps1(value[i]);//t3=value[i](4个value[i])
            t2=_mm_load_ps(c[row[i]]+k);//t2=c[row[i]]+k
            sum = _mm_mul_ps(t3,t1);//sum=t3*t1
            t2=_mm_add_ps(t2,sum);//t2+=t3*t1=value[i]*b[col[i]][k]
            _mm_store_ps(c[row[i]]+k,t2);//c[row[i]][k]=t2
        }
        for(int k=n-choice;k < n;k++){
            c[row[i]][k] += value[i] * b[col[i]][k];
        }
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}


//初始化
void init(){
    nonzeros=0;
    nozerorows=0;
    // THREAD_NUM=4;
    srand((int)time(0));
    s=0.005;
    y=new float[n]{0};
    yy=new float[n]{0};
    yyy=new float[n]{0};
    generate_vector(n,vec);//生成向量
    generate_sparse_matrix(mat,n,s);//生成稀疏矩阵mat
    generate_matrix(mat_nonsparse,n);//生成稠密矩阵
    mat_res1=new float*[n];
    mat_res2=new float*[n];
    mat_res3=new float*[n];
    for(int i=0;i<n;i++)
    {
        mat_res1[i]=new float[n]{0};
        mat_res2[i]=new float[n]{0};
        mat_res3[i]=new float[n]{0};
    }
    nonzeros=matrix_to_coo(mat,n,value,row,col,index);//生成对应的COO表示的稀疏矩阵
    single_circle=nozerorows/(THREAD_NUM*100);
}




int next_arr = 0;
pthread_mutex_t  mutex_task;

//实现pThread的spmm算法
///这里线程有两种划分模式，一种是直接在外层进行划分，另一种是在内层进行划分
///第一种实现的是在外层进行划分
void* coo_multiply_matrix_pthread1(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;//线程号
    int interval=nozerorows/THREAD_NUM;//间隔的设定
    int maxx=0;
    if(id==3){
        maxx=nonzeros;

    }else{
        maxx=index[interval*(id+1)];
    }

    for(int i=index[interval*id];i<maxx;i++){//计算的index[]是从row的i行到i+interval行
        for(int k=0;k<n;k++)
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    }//外层划分
    pthread_exit(NULL);
}
///第二种方法是在内层进行划分
void* coo_multiply_matrix_pthread2(void *parm){
    threadParm_t2 *p = (threadParm_t2 *) parm;
    int id = p->threadId;
    int i=p->rowid;
    int interval=n/THREAD_NUM;//间隔的设定
    int maxx=0;
    if(id==3){
        maxx=n;

    }else{
        maxx=interval*(id+1);
    }

    //for(int i=index[interval*id];i<maxx;i++){//计算的index[]是从row的i行到i+interval行
    for(int k=interval*id;k<maxx;k++)
        mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    //}
    pthread_exit(NULL);
}

int next_arr2 = 0;
//实现pThread编程的spmm动态线程分配
void* coo_multiply_matrix_pthread4(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;//线程id
    __m128 t1,t2,t3,sum;
    int choice = n % 4;
    int task = 0;//任务数
    int maxx;
    while(1){
        pthread_mutex_lock(&mutex_task);
        task = next_arr2;
        next_arr2+=single_circle;
        pthread_mutex_unlock(&mutex_task);
        if (task >= nozerorows) break;
        if(task>=nozerorows-single_circle)maxx=nonzeros;
        else maxx=index[task+single_circle];
        for(int i=index[task];i<maxx;i++){
            for(int k=0;k<n-choice;k+=4)//一次求4组
            {
                t1=_mm_load_ps(mat_nonsparse[col[i]]+k);
                sum = _mm_setzero_ps();
                t3 = _mm_set_ps1(value[i]);
                t2=_mm_load_ps(mat_res1[row[i]]+k);
                sum = _mm_mul_ps(t3,t1);
                t2=_mm_add_ps(t2,sum);
                _mm_store_ps(mat_res1[row[i]]+k,t2);
            }
            for(int k=n-choice;k < n;k++){
                mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
            }
        }
    }
    pthread_exit(NULL);
}



//实现pThread编程的spmm动态线程分配2
void* coo_multiply_matrix_pthread3(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int task = 0;
    int maxx;
    while(1){
        pthread_mutex_lock(&mutex_task);
        task = next_arr2;
        next_arr2+=single_circle;
        pthread_mutex_unlock(&mutex_task);
        if (task >= nozerorows) break;
        if(task>=nozerorows-single_circle)maxx=nonzeros;
        else maxx=index[task+single_circle];
        for(int i=index[task];i<maxx;i++){
            for(int k=0;k<n;k++)
                mat_res3[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }//多线程动态分配
    pthread_exit(NULL);
}

///实现pThread编程和sse编程结合的技术
void* coo_multiply_matrix_pthread_sse1(void *parm){
    threadParm_t *p = (threadParm_t *) parm;
    int id = p->threadId;
    int interval=nozerorows/THREAD_NUM;//间隔的设定
    int maxx=0;
    __m128 t1,t2,t3,sum;
    int choice = n % 4;
    if(id==3){
        maxx=nonzeros;//最大非零个数

    }else{
        maxx=index[interval*(id+1)];
    }

    for(int i=index[interval*id];i<maxx;i++){//计算的index[]是从row的i行到i+interval行
            //mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        for(int k=0;k<n-choice;k+=4)
        {
            t1=_mm_load_ps(mat_nonsparse[col[i]]+k);
            sum = _mm_setzero_ps();
            t3 = _mm_set_ps1(value[i]);
            t2=_mm_load_ps(mat_res1[row[i]]+k);
            sum = _mm_mul_ps(t3,t1);
            t2=_mm_add_ps(t2,sum);
            _mm_store_ps(mat_res1[row[i]]+k,t2);
        }
        for(int k=n-choice;k < n;k++){
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }
    pthread_exit(NULL);
}


//pThread实现spMM静态线程分配第一种算法封装
double spMM_pThread_static1(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);

    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread1, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//pThread实现spMM静态线程分配第二种算法封装------------------------这种算法造成的性能开销异常大，cache水平
double spMM_pThread_static2(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t2 threadParm[THREAD_NUM];
    for (int j=0;j<nonzeros;j++)
    {
        for (int i = 0; i < THREAD_NUM; i++)
        {
            threadParm[i].threadId = i;
            threadParm[i].rowid = j;
            pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread2, (void *)&threadParm[i]);
        }
        for (int i = 0; i < THREAD_NUM; i++)
        {
            pthread_join(thread[i], nullptr);
        }
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

//pThread实现spMM代码封装：动态线程分配函数
double spMM_pThread_dynamic1(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];

    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread3, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

double spMM_pThread_dynamic_sse(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];

    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread4, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}



//pThread实现spMM-SSE代码封装
double spMM_pThread_sse1(int thread_num){
    THREAD_NUM=thread_num;
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];

    for (int i = 0; i < THREAD_NUM; i++)
    {
      threadParm[i].threadId = i;
      pthread_create(&thread[i], nullptr, coo_multiply_matrix_pthread_sse1, (void *)&threadParm[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++)
    {
      pthread_join(thread[i], nullptr);
    }
    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}


///--------------------------------------------------------openMP编程中的spMM算法优化------------------------------------------
///实现spMM的openMP编程版本静态线程分配
double coo_multiply_matrix_openMP_static(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);

    int i,k;
    #pragma omp parallel for num_threads(OMP_NUM_THREADS),private(i, k)
    for (i=0;i<nonzeros;i++)//对非零元素循环
    {
        for(k=0;k<n;k++)//计算该元素所在行的影响
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

///实现spMM的openMP编程版本动态线程分配
double coo_multiply_matrix_openMP_dynamic(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);

    int i,k;
    #pragma omp parallel num_threads(OMP_NUM_THREADS),private(i, k)
    //#pragma omp for schedule(static, nozerorows/OMP_NUM_THREADS)dynamic, 50
    #pragma omp for schedule(guided)// guided动态线程分配
    for (i=0;i<nonzeros;i++)
    {
        for(k=0;k<n;k++)
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

///实现spMM的openMP编程版本静态线程分配+SSE并行
double coo_multiply_matrix_openMP_static_sse(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    __m128 t1,t2,t3,sum;
    int choice = n % 4;
    int i,k;
    #pragma omp parallel for num_threads(OMP_NUM_THREADS),private(i, k,t1,t2,t3,sum)
    for (i=0;i<nonzeros;i++)
    {
        for(int k=0;k<n-choice;k+=4)//循环内部使用SSE并行
            {
                t1=_mm_load_ps(mat_nonsparse[col[i]]+k);
                sum = _mm_setzero_ps();
                t3 = _mm_set_ps1(value[i]);
                t2=_mm_load_ps(mat_res1[row[i]]+k);
                sum = _mm_mul_ps(t3,t1);
                t2=_mm_add_ps(t2,sum);
                _mm_store_ps(mat_res1[row[i]]+k,t2);
            }
        for(int k=n-choice;k < n;k++){
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}

///实现spMM的openMP编程版本动态线程分配+SSE并行
double coo_multiply_matrix_openMP_dynamic_sse(){
    struct timeval val,newVal;
    gettimeofday(&val, NULL);
    __m128 t1,t2,t3,sum;
    int choice = n % 4;
    int i,k;
    #pragma omp parallel num_threads(OMP_NUM_THREADS),private(i, k,t1,t2,t3,sum)
    //#pragma omp for schedule(static, nozerorows/OMP_NUM_THREADS)dynamic, 50
    #pragma omp for schedule(guided)//动态分配
    for (i=0;i<nonzeros;i++)
    {
        for(int k=0;k<n-choice;k+=4)
            {
                t1=_mm_load_ps(mat_nonsparse[col[i]]+k);
                sum = _mm_setzero_ps();
                t3 = _mm_set_ps1(value[i]);
                t2=_mm_load_ps(mat_res1[row[i]]+k);
                sum = _mm_mul_ps(t3,t1);
                t2=_mm_add_ps(t2,sum);
                _mm_store_ps(mat_res1[row[i]]+k,t2);
            }
        for(int k=n-choice;k < n;k++){
            mat_res1[row[i]][k] += value[i] * mat_nonsparse[col[i]][k];
        }
    }

    gettimeofday(&newVal, NULL);
    double diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
	return diff / Converter;
}


///1.对比spMM几种算法的性能
///1.改变矩阵规模测试：100——10000
///2.改变线程数目：4——10
///3.改变稀疏度：0.001——0.05
///4.动态线程改变不同矩阵规模下的单个任务的单位:基本确定了矩阵进行计算的单位
void spMM_all_test(){
    init();
    double serial,spmm_static1,spmm_dynamic,spmm_dynamic_sse,spmm_dynamic_avx,spmm_sse,spmm_avx,spmm_pthread_sse,spmm_pthread_avx,
        spmm_openmp_static,spmm_openmp_dynamic,spmm_openmp_dynamic_sse,spmm_openmp_dynamic_avx,spmm_openmp_static_sse,spmm_openmp_static_avx = 0;
    //串行
    serial=coo_multiply_matrix_serial(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
    //SSE
    spmm_sse=coo_multiply_matrix_sse(nonzeros,n,row,col,value,mat_nonsparse,mat_res2);
    //pThread
    spmm_static1=spMM_pThread_static1(THREAD_NUM);
    spmm_dynamic=spMM_pThread_dynamic1(THREAD_NUM);
    spmm_pthread_sse=spMM_pThread_sse1(THREAD_NUM);
    next_arr2 = 0;
    spmm_dynamic_sse=spMM_pThread_dynamic_sse(THREAD_NUM);
    next_arr2 = 0;
    //openMP
    OMP_NUM_THREADS=THREAD_NUM;
    spmm_openmp_static=coo_multiply_matrix_openMP_static();
    spmm_openmp_static_sse=coo_multiply_matrix_openMP_static_sse();

    next_arr2 = 0;
    next_arr = 0;
    spmm_openmp_dynamic=coo_multiply_matrix_openMP_dynamic();
    next_arr2 = 0;
    next_arr = 0;
    spmm_openmp_dynamic_sse=coo_multiply_matrix_openMP_dynamic_sse();
    next_arr2 = 0;
    next_arr = 0;

    cout<<"矩阵规模:"<<n << "\t线程数:"<<THREAD_NUM<<endl
        <<"算法:                "<<"运行时间   "<<"加速比"<<endl
        <<"串行:                "<<serial<< "   " <<"100%"  <<endl
        <<"SIMD:               "<<spmm_sse<<"   " <<serial/spmm_sse*100 <<"%" << endl
        <<"pThread静态分配:     "<<spmm_static1<<"   " <<spmm_static1/spmm_sse*100 <<"%"                           <<endl
        <<"pThread动态分配:     "<<spmm_dynamic<<"   " <<spmm_dynamic/spmm_sse*100 <<"%"                           <<endl
        <<"pThread静态分配+SIMD:"<<spmm_pthread_sse<<"   " <<spmm_pthread_sse/spmm_sse*100 <<"%"                         <<endl
        <<"pThread动态分配+SIMD:"<<spmm_dynamic_sse<<"   " <<spmm_dynamic_sse/spmm_sse*100 <<"%"                       <<endl
        <<"openMP静态分配:      "<<spmm_openmp_static<<"   " <<spmm_openmp_static/spmm_sse*100 <<"%"                    <<endl
        <<"openMP动态分配:      "<<spmm_openmp_dynamic <<"   " <<spmm_openmp_dynamic/spmm_sse*100 <<"%"                <<endl
        <<"openMP静态分配+SIMD: "<<spmm_openmp_static_sse <<"   " <<spmm_openmp_static_sse/spmm_sse*100 <<"%"              <<endl
        <<"openMP动态分配+SIMD: "<<spmm_openmp_dynamic_sse <<"   " <<spmm_openmp_dynamic_sse/spmm_sse*100 <<"%"                <<endl;
        
}


int main()
{

    THREAD_NUM=4;
    n=4096;
    spMM_all_test();

    THREAD_NUM=8;
    n=4096;
    spMM_all_test();

    THREAD_NUM=16;
    n=4096;
    spMM_all_test();
    //释放内存空间
    delete []mat;
    delete []mat_nonsparse;
    delete []mat_res1;
    delete []mat_res2;
    delete []vec;
    delete []y;
    delete []yy;
    delete []value;
    delete []col;
    delete []row;
    delete []index;
    return 0;
}


