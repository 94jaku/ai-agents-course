import OpenAI from "openai";

const client = new OpenAI();

const tools: OpenAI.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "calculate",
      description: "Perform basic math",
      parameters: {
        type: "object",
        properties: {
          expression: { type: "string", description: "Math expression like '2 + 2'" },
        },
        required: ["expression"],
      },
    },
  },
];

function calculate(expression: string): number {
  return eval(expression);
}

async function main() {
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "user", content: "What is 25 * 4 + 10?" },
  ];

  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages,
    tools,
  });

  const msg = response.choices[0].message;
  messages.push(msg);

  if (msg.tool_calls) {
    for (const toolCall of msg.tool_calls) {
      const args = JSON.parse(toolCall.function.arguments);
      const result = calculate(args.expression);
      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: String(result),
      });
    }

    const finalResponse = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
    });

    console.log(finalResponse.choices[0].message.content);
  } else {
    console.log(msg.content);
  }
}

main();
