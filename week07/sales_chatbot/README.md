# 销售机器人

## 运行说明
0. 设置环境变量
```
export OPENAI_API_KEY=<change to your openai key>
```
1. 执行数据加载任务, 将txt的数据文件加载并保存成向量数据库格式的目录中,执行以下命令:
```
python3 sales_data_loader.py 
```
1. 运行web项目,执行以下命令:
```
python3 sales_chatbot.py
```

## 配置说明
项目的可以通过`conf.yaml`进行配置, 其中多个机器人可以通过bots配置进行扩展, 配置说明如下:
```yaml
# 机器人配置
bots:
  - name: estates # 机器人名称, 统一使用英文, 不能重复
    title: 房产销售 # 界面显示的标题
    vector_store_dir: real_estates_sales # 加载知识库的向量数据库目录
    score_threshold: 0.8 # 过滤知识库的相似度
    data_file: real_estate_sales_data.txt # 加载知识库的文本文件
```
