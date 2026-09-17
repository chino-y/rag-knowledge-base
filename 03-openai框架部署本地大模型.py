from openai import OpenAI
apikey="1231231"
baseurl="http://localhost:11434/v1"
model_name="deepseek-r1:1.5b" #必须是你已安装的模型才可以访问

client = OpenAI(
    api_key=apikey,
    base_url=baseurl)

# 对话推理
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "介绍你自己"},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

print(response.choices[0].message.content)
