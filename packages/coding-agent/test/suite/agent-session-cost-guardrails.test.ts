import type { AgentEvent } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import { createHarness, type Harness } from "./harness.js";

type SessionCostInternals = {
	_processAgentEvent: (event: AgentEvent) => Promise<void>;
};

function providerFailure(kind: string, errorMessage: string, retryAfterMs?: number): AssistantMessage {
	return {
		...fauxAssistantMessage("", { stopReason: "error", errorMessage }),
		diagnostics: [
			{
				type: "provider_stream_failure",
				timestamp: Date.now(),
				details: { kind, ...(retryAfterMs !== undefined ? { retryAfterMs } : {}) },
			},
		],
	};
}

function costlyMessage(text: string, costUsd: number): AssistantMessage {
	const message = fauxAssistantMessage(text);
	return {
		...message,
		usage: {
			input: 100,
			output: 100,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 200,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: costUsd },
		},
	};
}

describe("AgentSession quota and cost guardrails", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("fails closed on structured quota exhaustion instead of retrying", async () => {
		const harness = await createHarness({ settings: { retry: { enabled: true, maxRetries: 3, baseDelayMs: 1 } } });
		harnesses.push(harness);
		const retryEvents: string[] = [];
		harness.session.subscribe((event) => {
			if (event.type === "auto_retry_start") retryEvents.push(`start:${event.attempt}`);
		});

		harness.setResponses([providerFailure("quota", "Provider usage quota or credit exhausted (429)")]);

		await harness.session.prompt("test");

		expect(retryEvents).toEqual([]);
		expect(harness.faux.state.callCount).toBe(1);
	});

	it("fails closed on quota language in the error message without structured diagnostics", async () => {
		const harness = await createHarness({ settings: { retry: { enabled: true, maxRetries: 3, baseDelayMs: 1 } } });
		harnesses.push(harness);
		const retryEvents: string[] = [];
		harness.session.subscribe((event) => {
			if (event.type === "auto_retry_start") retryEvents.push(`start:${event.attempt}`);
		});

		harness.setResponses([
			fauxAssistantMessage("", {
				stopReason: "error",
				errorMessage: "429: you have exceeded your premium request quota",
			}),
		]);

		await harness.session.prompt("test");

		expect(retryEvents).toEqual([]);
		expect(harness.faux.state.callCount).toBe(1);
	});

	it("keeps retrying plain rate limits and honors the provider's Retry-After for the delay", async () => {
		const harness = await createHarness({ settings: { retry: { enabled: true, maxRetries: 3, baseDelayMs: 1 } } });
		harnesses.push(harness);
		const delays: number[] = [];
		harness.session.subscribe((event) => {
			if (event.type === "auto_retry_start") delays.push(event.delayMs);
		});

		harness.setResponses([
			providerFailure("rate_limit", "Provider rate limit exceeded (429)", 5),
			fauxAssistantMessage("recovered"),
		]);

		await harness.session.prompt("test");

		expect(delays).toEqual([5]);
		expect(harness.faux.state.callCount).toBe(2);
	});

	it("fails fast when Retry-After exceeds retry.provider.maxRetryDelayMs", async () => {
		const harness = await createHarness({ settings: { retry: { enabled: true, maxRetries: 3, baseDelayMs: 1 } } });
		harnesses.push(harness);
		const retryEvents: Array<{ type: string; finalError?: string }> = [];
		harness.session.subscribe((event) => {
			if (event.type === "auto_retry_start") retryEvents.push({ type: "start" });
			if (event.type === "auto_retry_end") retryEvents.push({ type: "end", finalError: event.finalError });
		});

		harness.setResponses([providerFailure("rate_limit", "Provider rate limit exceeded (429)", 120_000)]);

		await harness.session.prompt("test");

		expect(retryEvents).toHaveLength(1);
		expect(retryEvents[0].type).toBe("end");
		expect(retryEvents[0].finalError).toContain("maxRetryDelayMs");
		expect(harness.faux.state.callCount).toBe(1);
	});

	it("still waits for Retry-After when maxRetryDelayMs is 0 (cap disabled)", async () => {
		const harness = await createHarness({
			settings: {
				retry: { enabled: true, maxRetries: 3, baseDelayMs: 1, provider: { maxRetryDelayMs: 0 } },
			},
		});
		harnesses.push(harness);
		const delays: number[] = [];
		harness.session.subscribe((event) => {
			if (event.type === "auto_retry_start") delays.push(event.delayMs);
		});

		harness.setResponses([
			providerFailure("rate_limit", "Provider rate limit exceeded (429)", 5),
			fauxAssistantMessage("recovered"),
		]);

		await harness.session.prompt("test");

		expect(delays).toEqual([5]);
		expect(harness.faux.state.callCount).toBe(2);
	});

	it("stops the session and refuses subagent spawns at the cost ceiling", async () => {
		const harness = await createHarness({
			settings: { budget: { maxSessionCostUsd: 5 } },
			rlmDepth: 0,
			rlmMaxDepth: 1,
		});
		harnesses.push(harness);
		const limitEvents: Array<{ totalUsd: number; limitUsd: number }> = [];
		harness.session.subscribe((event) => {
			if (event.type === "cost_limit_reached") {
				limitEvents.push({ totalUsd: event.totalUsd, limitUsd: event.limitUsd });
			}
		});

		// The faux provider synthesizes zero-cost usage, so feed costed assistant
		// turns through the event pipeline directly.
		const internals = harness.session as unknown as SessionCostInternals;
		await internals._processAgentEvent({
			type: "message_end",
			message: costlyMessage("expensive answer", 3),
		} as AgentEvent);
		expect(harness.session.isCostLimited).toBe(false);
		expect(limitEvents).toEqual([]);

		await internals._processAgentEvent({
			type: "message_end",
			message: costlyMessage("second expensive answer", 3),
		} as AgentEvent);

		expect(limitEvents).toEqual([{ totalUsd: 6, limitUsd: 5 }]);
		expect(harness.session.isCostLimited).toBe(true);
		await expect(harness.session.runRlmChild("blocked child")).rejects.toThrow(/Session cost limit reached/);
	});

	it("stays under the ceiling for cheap turns and does not emit the limit event", async () => {
		const harness = await createHarness({ settings: { budget: { maxSessionCostUsd: 5 } } });
		harnesses.push(harness);
		const limitEvents: unknown[] = [];
		harness.session.subscribe((event) => {
			if (event.type === "cost_limit_reached") limitEvents.push(event);
		});

		const internals = harness.session as unknown as SessionCostInternals;
		await internals._processAgentEvent({
			type: "message_end",
			message: costlyMessage("cheap answer", 0.01),
		} as AgentEvent);

		expect(limitEvents).toEqual([]);
		expect(harness.session.isCostLimited).toBe(false);
	});

	it("caps concurrent subagent spawns at rlmMaxChildren", async () => {
		const harness = await createHarness({
			settings: { rlmMaxChildren: 1 },
			rlmDepth: 0,
			rlmMaxDepth: 1,
		});
		harnesses.push(harness);

		const internals = harness.session as unknown as { _activeRlmChildRuns: Map<string, unknown> };
		internals._activeRlmChildRuns.set("occupied", {});

		await expect(harness.session.runRlmChild("one too many")).rejects.toThrow(/concurrent subagent limit reached/);
	});
});
