Break the Agent
by AfterQuery

The Challenge
Build a reinforcement learning environment that challenges today's best AI agents, demonstrate real failure modes, and show that its rewards reflect real progress.

Introduction
Today's cutting-edge AI systems can solve Millennium Problems, discover new drugs, and create entire production apps with a few lines of prompting. But they still struggle in a wide variety of novel situations, ranging from everyday use cases to niche, highly technical workflows. We ask: what should AI agents be able to do that they still struggle with today?

At AfterQuery, we build environments for training and testing agents. The work starts with choosing a problem worth learning, defining an action and observation space, and defining a reward function that defines what success looks like. Getting those pieces right is a heavy research and engineering problem in itself.

For this challenge, we want to see your take on it. Build a world with something interesting to learn. Find out where strong agents struggle, and whether your reward actually encourages the behavior you intended.

The challenge
Choose a setting you find interesting and build an environment an agent can interact with. It could be a game, a scientific simulation, a software task, a real-world process, or something we haven't thought of. Explain why you chose it and what an agent could learn there.

Your environment should have a clear goal, actions the agent can take, and a reward you can check. Run a frontier agent in it and compare the results with a stronger reference run your team can demonstrate and reproduce. We'll provide the agent harness and model credits.

Show where the agent falls short and why. The challenge should come from the decisions it needs to make, with access to the information and tools needed to succeed. We'll also try to find shortcuts that earn a high score without completing the task: reward hacking.

The setting and approach are yours to choose. We care about what makes the environment useful for learning and what your experiments reveal.

Optional stretch goal
Train a smaller model using experience from your environment and show whether it improves.

Prizes
The top three teams receive €3,000, €1,500 and €500 in cash. Members of the winning teams can go straight to the final stage of AfterQuery's interview process: a work trial with our environments team, when they're ready. Finalists also receive AfterQuery gear.

How we judge: We'll look at four things: how much room there is for the agent to improve, whether its mistakes come from meaningful challenges, how well the reward holds up to exploits, and how well the environment is built and the results can be reproduced.

Judging rubric
• The gap (30 p) – Full points: the frontier agent consistently scores below a stronger reference run your team demonstrates; both results can be reproduced across repeated runs. Zero points: no measurable gap, or the claimed gap cannot be reproduced or checked against a demonstrated reference run. • Meaningful difficulty (30 p) – Full points: you can explain the agent's main mistakes and connect them to decisions in the task; the information and tools needed to do better are available to it. Zero points: the gap comes from unfairly withheld information, trick wording, impossible timing, or broken tools. • Reward integrity (25 p) – Full points: the reward measures success at the task and holds up when judges try to exploit it; doing nothing or stopping early earns little or no credit; completing the task well remains the best way to score. Zero points: an exploit, scoring loophole, or strategy that avoids the task earns a better score than successful play. • Craft (15 p) – Full points: the environment's rules and behavior make sense for the setting; judges can run the evaluation from your README with one command and reproduce your reported results. Zero points: the environment only works in your demo, or judges cannot reproduce the evaluation.

How judging runs: Each team gets a 10-minute demo. Judges then rerun the evaluation with the standard harness: your reference run, three agent runs, a do-nothing baseline, and a stop-early baseline. They also spend 15 minutes looking for reward exploits. Scores are based on these runs.

Tiebreaker: If point totals are close, we'll favor a team that shows a smaller model improving after training on experience from its environment.

Insight
Starting points
Planning several steps ahead

Exploring an unfamiliar system

Using tools

Managing limited resources

Adapting when new information arrives

These are prompts, not a list of required features.

A small environment with a well-chosen challenge can be enough. Think about what better play looks like, how you would measure it, and whether an agent could increase its score without getting better at the task.

How judging runs
Each team gets a 10-minute demo. Judges then rerun the evaluation with the standard harness: your reference run, three agent runs, a do-nothing baseline, and a stop-early baseline. They also spend 15 minutes looking for reward exploits. Scores are based on these runs.

Tiebreaker: if point totals are close, we'll favor a team that shows a smaller model improving after training on experience from its environment.

Resources
What we'll bring
An open-source agent harness – Harbor running the OpenCode agent – that connects to your environment over MCP, a scoring skeleton, and frontier model API credits for each team to benchmark throughout the 48 hours.

Open-source agent harness: Harbor + OpenCode
Scoring skeleton
Frontier model API credits per team
On-site mentors for environment design reviews and reward testing
Starter kit: harness documentation and the judging rubric
Mentors
Daksh Raghuwanshi and Ved Vedere from AfterQuery's environments team, on site all weekend. They build RL environments and agent evaluations for frontier AI labs.

Company Info

AfterQuery
Website: https://www.afterquery.com/

Judging Criteria
The gap
(30%)
Full points: the frontier agent consistently scores below a stronger reference run your team demonstrates; both results can be reproduced across repeated runs. Zero points: no measurable gap, or the c

Meaningful difficulty
(30%)
Full points: you can explain the agent's main mistakes and connect them to decisions in the task; the information and tools needed to do better are available to it. Zero points: the gap comes from unf

Reward Integrity
(25%)
Full points: the reward measures success at the task and holds up when judges try to exploit it; doing nothing or stopping early earns little or no credit; completing the task well remains the best wa

Craft
(15%)
Full points: the environment's rules and behavior make sense for the setting; judges can run the evaluation from your README with one command and reproduce your reported results. Zero points: the envi

Prizes
1st Place
3,000€ + A work trial at AfterQuery + Merch

2nd Place
1,500€ + Merch

3rd Place
500€


The Challenge
INTRODUCTION
From electron to token, it takes decades of progress to run AI. Add scale, efficiency, and sustainability, and AI becomes a monumental achievement.

Verda brings everything together.

We make compute, storage, and networking work in unison to squeeze peak performance, hosted in European data centers and utilizing 100% renewable energy. We develop services for you to utilize computational power most efficiently, no matter your use case.

The end result is a cloud platform trusted by the world's leading AI labs and frontier researchers. The best thing is you get to use the same next-gen compute for free at Junction 2026.

BOUNTY: BEST USE OF VERDA
Pick any challenge at Junction and use Verda products to build something you couldn't build on a laptop. Then show us how far it goes.

Teach a humanoid to move. Train locomotion or manipulation policies for Aalto's Unitree G1 in simulation, at scale, before your shot at the real robot.
Turn demonstrations into a brain. Train imitation-learning or VLA policies from robot-arm demos in the Ignite × Intel challenge, or from Rokoko motion-capture teleoperation data.
Make an agent stronger than it was on Friday. Run thousands of rollouts in your AfterQuery RL environment and fine-tune a smaller model on the experience.
Read the whole internet's encyclopedia. Embed MariaDB's full Wikipedia dataset with batch jobs and serve a RAG assistant on top.
Keep sensitive data in Europe. Run open-weight models for a banking coach or a recruitment tool on infrastructure you control, instead of sending data to someone else's API.
Or surprise us. Anything that needs serious compute counts. (Anything except crypto mining. We've seen it, and it gets you banned.)

What we're looking for

An ambitious idea where compute is what makes it possible
The right Verda product for the job, and a clear reason why you chose it
Your path from first idea to working prototype
Bonus points: show how your solution would scale from a hackathon demo to real users
Minimum Requirements

Document your process
Submit via the Junction Platform > Challenges > Verda Compute Track
The Technology

Inference API for serving out-of-the-box open-weight models within your application.
GPU instances or Instant clusters for training or fine-tuning tasks.
CPU instances for hosting agentic applications.
Auto-scaling containers for hosting models where scale and costs matter.
Batch jobs for long-running tasks in the background.
Prizes and Judging

The winning team gets 10,000 USD in cloud credits and Verda hoodies. Runner-up teams in second and third places get 2,500 USD in cloud credits.

Resources
What we'll bring
Inference API: serve open-weight models in your app with zero setup
GPU instances and Instant Clusters: for training and fine-tuning when one GPU isn't enough
CPU instances: to host your agents and backends
Auto-scaling containers: deploy models where scale and cost matter
Batch jobs: run long jobs in the background while you sleep (or don't)
NVIDIA Nemotron endpoint: hosted by Verda together with NVIDIA
Free credits for every team: scan the QR code on your table or go to verda.com/blog/junction-2026
Mentors on-site during the entire weekend, meet us on the third floor at Booth 6! Start with the "Getting started with Verda" workshop on Friday at 20:30, right after the opening ceremony: sign-up, credits, every product, and how to use them.

Ambition: how much did compute let you do that you otherwise couldn't? Fit: did you pick the right Verda services and use them well? Working prototype: does it run, and is the process documented? Path to scale: is there a credible way from demo to real users?

Company Info

Verda
Website: http://verda.com/

Judging Criteria
Ambition
(25%)
How much did compute let you do that you otherwise couldn't?

Fit
(25%)
Did you pick the right Verda services and use them well?

Working prototype
(25%)
Does it run, and is the process documented?

Path To Scale
(25%)
Is there a credible way from demo to real users?

Prizes
1st Place
$10,000 in Verda cloud credits + Verda hoodies

2nd Place
$2,500 in cloud credits

3rd Place

The Challenge
OpenAI challenges you to push the boundaries of what's possible with AI-powered software development.

Use Codex as part of your development process to build, iterate, and ship an ambitious project during Junction 2026.

Whether you're building AI agents, complex applications, or entirely new products, show us how Codex can accelerate your development and help you achieve more in just 48 hours.

This is an open challenge: You can participate alongside any other Junction challenge. The key is demonstrating how you've meaningfully used Codex in building your solution.

Resources
OpenAI Codex: https://developers.openai.com/codex

Codex documentation: https://developers.openai.com/codex/quickstart

OpenAI developer platform: https://platform.openai.com/docs

Exclusive Junction 2026 offer: OpenAI is providing $100 in credits per participant through unique redemption links distributed by Junction. The redemption links can be found here.

Company Info

OpenAI
OpenAI is a leading artificial intelligence research and deployment company focused on building safe, powerful AI systems that expand human capabilities. The organization prioritizes alignment, reliability, and real-world impact, ensuring that advanced models benefit individuals, companies, and society at scale. With world-class research talent, robust compute infrastructure, and cutting-edge model APIs, OpenAI empowers developers to build transformative applications. OpenAI takes part in Junction to support frontier innovation, collaborate with ambitious builders, and accelerate the creation of practical, high-impact AI solutions.

Website: https://openai.com/

Prizes
1st Overall
$10 000 API Credits

2nd Overall
$5000 API Credits

3rd Overall
$2500 API Credits

Best Use of Gemma
by Google DeepMind

The Challenge
Awarded to the team that demonstrates the most innovative and impactful use of Google's Gemma open models. We're looking for creative applications, strong technical execution, and solutions that showcase the potential of open AI models to solve real-world problems.

Resources
How to Access Gemma
Gemma is Google's family of open AI models, available for developers to build with.

You can access and run Gemma through several platforms, including our Junction 2026 partners:

Featherless AI – Access hosted Gemma models through an API.
Modal – Deploy and run Gemma models on serverless GPU infrastructure.
Verda – Use cloud GPU resources to deploy and run Gemma yourself.
Google AI Studio – Experiment with Gemma in your browser.
Hugging Face – Download Gemma models and integrate them into your project.
For setup instructions and technical resources, check out the official Gemma documentation.

Be sure to check the partner resources available in the Junction Hackerpack for any event-specific credits or access instructions.

Company Info

Google DeepMind
Website: https://deepmind.google/

Prizes
Top 3
€1,000 per team for the top 3 teams, with no ranking between winners.

Best Use of Nemotron
by NVIDIA

The Challenge
Build a project powered by NVIDIA Nemotron that solves a real problem through reasoning, tool use, or coordinated workflows. Applies to: Any track.

INTRODUCTION
What can you build when AI moves beyond answering questions and starts getting things done? NVIDIA invites you to use Nemotron to build a project that tackles a real problem for a clear audience. Connect it to tools, ground it in useful data, or combine it with other models to turn an idea into a working solution. Show us what Nemotron enables and why it matters to the people using your project.

THE TECHNOLOGY
Meet Nemotron 3.5 Super VL, NVIDIA's newest open model, and it's yours to build with this weekend. It has 120B parameters but only wakes up 12B per token, so it runs fast for its size, and it can read text, images, documents and even video in a single workflow. That means your agent can look at a screenshot, read a PDF, watch a clip, call a tool and write the code to act on it, all with one model.

THE CHALLENGE
Build a project where Nemotron plays a meaningful role in doing real work. Use it on its own or alongside other models, tools, and data sources to solve a problem for a clear audience. Nemotron must contribute directly to the project's core functionality, and your demo should make that contribution visible. The solution is open-ended: choose the problem, design the workflow, and show what your approach enables.

Minimum Requirements:

Use NVIDIA Nemotron 3.5 Super VL model in the working solution.
Demonstrate its contribution to the project's core functionality.
Submit a short explanation identifying the model used, what it does, why it matters, and how you improved its outputs.
This should be done as text input into the submission form.
GETTING STARTED
NVIDIA Build: Free hosted APIs to explore, test, and prototype with NVIDIA and community AI models.
Note: The API is OpenAI-compatible with base URL https://integrate.api.nvidia.com/v1 (for example, model "nvidia/nemotron-3.5-lightning-30b-a3b"). The free tier is rate-limited (developer forums report about 40 requests per minute per key), so agent loops should use retries with backoff and each teammate should use their own key.
OpenRouter API: Serverless path with a unified API across providers.
Note: Free variants (ending in :free) are capped at 20 requests per minute and 50 requests per day, or 1,000 per day once the account has bought at least 10 credits.
Hugging Face: Open weights for local runs, fine-tuning or deployment on Brev.
RESOURCES AND SUPPORT
Support Channel: Nvidia Discord channel and at the NVIDIA table
PRIZES AND JUDGING
PRIZE
2 winners of 1x RTX 5080

JUDGING CRITERIA
Meaningful Nemotron Usage: Nemotron is central to the project's value. The team clearly explains what it does and why it matters.
Technical Execution: The demo works reliably and shows thoughtful implementation choices around architecture, data flow, tool use, latency, or error handling.
Quality of AI Output: Outputs are useful, relevant, and grounded where appropriate. The team demonstrates efforts to improve quality through prompting, evaluation, or feedback loops.
Creativity and Differentiation: The team uses Nemotron thoughtfully or inventively, with a distinct approach to the problem.
COMPANY AND CONTACTS
About NVIDIA
NVIDIA builds accelerated computing technologies and AI software that help developers turn ideas into applications. At Junction, we're inviting teams to explore NVIDIA Nemotron open models and build useful agents around real-world needs. Start with a problem you care about and show what you can do over the weekend.

Contacts
Main Contact: Elliot Winch, Community Program Manager, EMEA, ewinch@nvidia.com
Technical Contact:
Mentors Attending: Maxime Pariente and Ryan Reese
Judges: NVIDIA Team
Company Info

NVIDIA
Website: https://www.nvidia.com/en-us/

Judging Criteria
Meaningful Nemotron Usage
(25%)
Nemotron is central to the project's value. The team clearly explains what it does and why it matters.

Technical Execution
(25%)
The demo works reliably and shows thoughtful implementation choices around architecture, data flow, tool use, latency, or error handling.

Quality of AI Output
(25%)
Outputs are useful, relevant, and grounded where appropriate. The team demonstrates efforts to improve quality through prompting, evaluation, or feedback loops.

Creativity and Differentiation
(25%)
The team uses Nemotron thoughtfully or inventively, with a distinct approach to the problem.

Prizes
Two best teams
1x NVIDIA RTX 5080

All Challenges
/
Best Use of Context.dev: Fully Structured Web
Status: Submissions are open
T -15:40:01
Best Use of Context.dev: Fully Structured Web
by Context.dev

The Challenge
Plug the Context.dev API into whatever you're building and give it the live web: pages, brands, companies, people and news as clean structured data in one API call.

The Challenge
Whatever you're building, use the Context.dev API to extract exactly what you need from the web and nothing more.

Live web data lets your project check facts against current sources, react when a page or news story changes, brand itself from a customer's email, or enrich a company or person in one call instead of relying on stale data.

To count, your project must call at least one Context.dev endpoint in its core flow, not just as a side feature.

Minimum requirements
Use at least one Context.dev API endpoint in the submitted project.
The data from Context.dev must visibly drive something in the demo.
What to submit
Submit a GitHub repository and a demo of your project, plus a short note on which Context.dev endpoints you used and why.

Resources
What we'll bring
Every team gets 50,000 Context.dev API credits that don't expire, plus documentation, SDKs, an MCP server, live example projects and Context.dev engineers on site to help.

No credit card needed. There is no rate limit, with a maximum of 50 concurrent requests per organisation. No special hardware is needed; the live web is your dataset.

The Technology
Context.dev is a web data API. You send it a URL, a domain, a company or a search query, and it returns clean, structured data — markdown, JSON, screenshots, logos, colours, company facts and news — that your app or model can use right away. No scrapers, proxies or parsers to maintain.

Key capabilities:

Search and scrape: turn pages into markdown, HTML, screenshots or images, or crawl and map a whole site.
Structured extraction: extract JSON with your own schema, or get researched answers to a question.
Brand data: get logos, colours, fonts and a style guide from a domain or email.
Search and enrichment: search and enrich companies, people and news.
Monitors and webhooks: react when a page or topic changes.
Batch processing: process thousands of URLs.
File parsing: parse PDFs and documents.
Social media: scrape feeds such as TikTok and Instagram.
Getting started
Create an account at context.dev/signup. Coding agents can use context.dev/auth.md.
Redeem code JUNCTION26 at Redeem credits to receive 50,000 API credits per team, with no expiry.
Copy a secret key from the API keys page.
Install the SDK: npm install context.dev or pip install context.dev.
Scrape your first page using the code in the Quickstart. You should see the Junction homepage printed as Markdown. Swap in any URL, or try brand, search, news or monitors. Start building!
Using a coding agent? Point it at the Agent quickstart, or connect the MCP server at https://mcp.context.dev/mcp (setup guide).

Example requests and responses:

Scrape the Junction website to Markdown: request + response.
Extract this event's details as JSON: request + response.
Recommended setup
Use any language over HTTP. SDKs are available for TypeScript/Node, Python, Ruby and Go. If you can call an API, you can use Context.dev, whether you're building a web app, backend or something else that can make an HTTP request.

Bring any laptop with Node.js 20+ or Python 3.9+ (Ruby 3.2+ and Go 1.22+ are also supported), or just curl. Internet access is required.

Optional tools: the context-dev CLI, the agent skill, or the MCP server. To try requests without code, use the Playground.

Tips and common pitfalls
Check each output's success flag. A 200 response can still contain a failed output.
Set maxAgeMs: 0 when you need fresh data. Results are cached for up to 3 days by default, and cached results cost the same credits as fresh ones. Store results in your own app to avoid paying for repeat calls.
Crawls, batches and monitors are asynchronous. Poll the job or receive a webhook; use a service such as Cloudflare Tunnel for local webhooks.
Some sites block automated access. Teams are responsible for respecting the terms of the sites they use.
Documentation and examples
Documentation: docs.context.dev
Guides: Quickstart, Agent quickstart, Recipes
API reference: Web scraping API
Operation costs: Credits and pricing
Live examples: GPU rental price tracker (source) and Yapper leaderboard
SDKs and integrations: github.com/context-dot-dev
There is no starter repository; start with the Quickstart guide.

Mentors and support
Context.dev engineers will help with web data extraction, search, LLM apps and large-scale data pipelines:

Beer van der Drift — CTO, challenge lead and lead judge
Yahia Bakour — Founder & CEO
Thomas Oomens — Engineer
On-site availability:

Friday: until 23:00
Saturday: from 09:00 until Sunday 02:00
Sunday: from 07:00 until the end
Come visit the Context.dev booth. Main and technical contact: Beer van der Drift, CTO, beer@context.dev. Service status: status.context.dev.

Company Info

Context.dev
Context.dev provides the web-data layer for AI: APIs that turn websites, brands, companies, people and news into structured, real-time context for software and agents. Teams like Supermemory, Mintlify and Upstash build on it. We went through YC S26 and raised a $9M seed led by General Catalyst, with investors like Paul Graham and Theo of T3. We're coming to Junction to see what the next generation of builders creates when the whole web is one API call away.

Judging Criteria
Creative use of Context.dev
(40%)
Unusual sources, combining several endpoints.

How central live web data is to the product
(30%)
Technical execution
(30%)
Accuracy, latency, handling of change over time.

Prizes
1st Place
€750 + swag

2nd Place
€250 + swag

3rd Place
Swag


Best use of Modal
by Modal

The Challenge
Take your Junction project from laptop to production AI on Modal.
Description
Every Junction team is building with AI, and most hit the same wall: the model runs on a laptop, but too slowly, not on a GPU, and not behind a URL the judges can open. Modal gets you past that wall. Use Modal to power the project you're already building for your main Junction challenge: serve an open model on a GPU, run your agent's code in secure Sandboxes, fine-tune a model on your own data, or fan out a big batch job, all in plain Python with no Dockerfiles, Kubernetes or YAML. We're looking for the most ambitious working AI system built on Modal, where Modal does the heavy lifting and the team spends its 48 hours on the problem. Inference, agents, sandboxes and training: your whole AI stack on one platform.

Mentors:
Engineers and Developer Relations from Modal with experience in inference, AI agents and code-execution sandboxes, fine-tuning and reinforcement learning, and real-time audio and video pipelines. We'll be at the Modal booth throughout the event and on the Junction Discord for debugging help.

Main contact person: Filip Makraduli, Developer Relations Engineer

Insight
Modal was built so that a single developer can do what used to take a platform team, which makes it a natural fit for a 48-hour hackathon. Best use of Modal is a tech challenge, so you compete in it on top of your main challenge: use Modal to power the project you're already building and you can enter. The same project can enter other tech challenges too.

What we’ll bring:
Every participant gets $100 in Modal credits for CPUs and GPUs, up to the latest NVIDIA B300s, billed per second and scaling to zero when idle. Modal is Python-native: decorate a function, run modal run, and it executes in the cloud in seconds; modal deploy turns it into an autoscaling web endpoint with a public URL ready for your demo. Teams can serve open-weight LLMs, vision-language and speech models; run untrusted, LLM-generated code in isolated Sandboxes; fine-tune or RL post-train models on one GPU or a multi-node cluster with Modal Dojo, our open-source training library; and store weights and datasets in Volumes. We'll provide a Junction starter kit of ready-to-run examples. The kick-off workshop will be at the Modal booth and announced on Discord, and Modal engineers will be on site all weekend.

Resources
Access and list of technologies we are bringing:
• Modal compute credits ($100 per participant) for CPUs and GPUs: T4, L4, A10, L40S, A100 (40 and 80 GB), RTX PRO 6000, H100, H200, B200 and B300 • Modal Python SDK and CLI (pip install modal), plus JavaScript and Go SDKs for Sandboxes • Modal Sandboxes: secure, on-demand containers for agents, code interpreters and evals • Web endpoints (FastAPI, WebSockets, WebRTC) with autoscaling and public URLs • Volumes, Dicts and Queues for model weights, datasets and shared state • Modal Dojo: open-source library for fine-tuning and RL post-training open models on Modal GPUs • Starter examples: LLM serving with vLLM and SGLang, speech-to-text, text-to-speech, vision, RAG, agents and RL • On-site mentors from Modal engineering and Developer Relations

To claim $100 in Modal credits you need to sign up for a Modal account, then redeem the credits via the link (https://modal.fillout.com/t/qMXCmRGseUus?code=VVX-2QX-8HW) using the code VVX-2QX-8HW.

The credits are valid for one year from redemption, so everyone can keep building well after the hackathon wraps up.

Company Info

Modal
Website: https://modal.com/

Judging Criteria
Effective use of Modal
(40%)
how central Modal is to the project and how well you use it, from GPU inference to Sandboxes, training and scaling out

Technical ambition
(25%)
how hard the build is and how far you pushed it in 48 hours

Impact
(20%)
how well it solves a real problem from your main challenge

Demo
(15%)
a live, deployed app the judges can actually use

Prizes
1st team
$