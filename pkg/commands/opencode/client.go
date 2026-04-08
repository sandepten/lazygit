package opencode

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	baseURL    string
	providerID string
	modelID    string
	httpClient *http.Client
}

func NewClient(baseURL, providerID, modelID string) *Client {
	return &Client{
		baseURL:    strings.TrimSuffix(baseURL, "/"),
		providerID: providerID,
		modelID:    modelID,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

func NewDefaultClient() *Client {
	return NewClient("http://localhost:4096", "github-copilot", "gpt-5.4-mini")
}

type Session struct {
	ID string `json:"id"`
}

type TextPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type PromptRequest struct {
	Model *ModelSpec `json:"model,omitempty"`
	Parts []TextPart `json:"parts"`
}

type ModelSpec struct {
	ProviderID string `json:"providerID"`
	ModelID    string `json:"modelID"`
}

type MessageResponse struct {
	Info  MessageInfo `json:"info"`
	Parts []Part      `json:"parts"`
}

type MessageInfo struct {
	ID   string `json:"id"`
	Role string `json:"role"`
}

type Part struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type SSEEvent struct {
	Type       string          `json:"type"`
	Properties json.RawMessage `json:"properties"`
}

func (c *Client) createSession() (*Session, error) {
	req, err := http.NewRequest("POST", c.baseURL+"/session", bytes.NewBuffer([]byte("{}")))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("failed to create session: status %d, body: %s", resp.StatusCode, string(body))
	}

	var session Session
	if err := json.NewDecoder(resp.Body).Decode(&session); err != nil {
		return nil, fmt.Errorf("failed to decode session response: %w", err)
	}

	return &session, nil
}

func (c *Client) sendMessageAsync(sessionID string, prompt string) error {
	requestBody := PromptRequest{
		Model: &ModelSpec{
			ProviderID: c.providerID,
			ModelID:    c.modelID,
		},
		Parts: []TextPart{
			{Type: "text", Text: prompt},
		},
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("%s/session/%s/message", c.baseURL, sessionID)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send message: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to send message: status %d, body: %s", resp.StatusCode, string(body))
	}

	return nil
}

func (c *Client) waitForIdleWithPolling(sessionID string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for response")
		case <-ticker.C:
			messages, err := c.getMessages(sessionID)
			if err != nil {
				continue
			}

			for _, msg := range messages {
				if msg.Info.Role == "assistant" {
					for _, part := range msg.Parts {
						if part.Type == "text" && part.Text != "" {
							return nil
						}
					}
				}
			}
		}
	}
}

func (c *Client) startEventListener(ctx context.Context, sessionID string) <-chan struct{} {
	idleChan := make(chan struct{}, 1)

	go func() {
		defer close(idleChan)

		req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/event", nil)
		if err != nil {
			return
		}
		req.Header.Set("Accept", "text/event-stream")

		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			return
		}
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			select {
			case <-ctx.Done():
				return
			default:
			}

			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")
			var event SSEEvent
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			if event.Type == "session.idle" {
				var props struct {
					SessionID string `json:"sessionID"`
				}
				if err := json.Unmarshal(event.Properties, &props); err == nil {
					if props.SessionID == sessionID {
						select {
						case idleChan <- struct{}{}:
						default:
						}
						return
					}
				}
			}
		}
	}()

	return idleChan
}

func (c *Client) getMessages(sessionID string) ([]MessageResponse, error) {
	url := fmt.Sprintf("%s/session/%s/message", c.baseURL, sessionID)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get messages: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("failed to get messages: status %d, body: %s", resp.StatusCode, string(body))
	}

	var messages []MessageResponse
	if err := json.NewDecoder(resp.Body).Decode(&messages); err != nil {
		return nil, fmt.Errorf("failed to decode messages: %w", err)
	}

	return messages, nil
}

func (c *Client) extractAssistantResponse(messages []MessageResponse) string {
	var result strings.Builder

	for _, msg := range messages {
		if msg.Info.Role == "assistant" {
			for _, part := range msg.Parts {
				if part.Type == "text" && part.Text != "" {
					result.WriteString(part.Text)
				}
			}
		}
	}

	return strings.TrimSpace(result.String())
}

func (c *Client) GenerateCommitMessage(stagedDiff string) (string, error) {
	if stagedDiff == "" {
		return "", fmt.Errorf("no staged changes to generate commit message for")
	}

	prompt := fmt.Sprintf(`Generate a concise git commit message for the following staged changes.
Follow conventional commit format (type: description).
Keep the summary line under 72 characters.
If there are multiple logical changes, you can add a brief body after a blank line.
Do not use any tools, just respond with plain text containing only the commit message:

%s`, stagedDiff)

	session, err := c.createSession()
	if err != nil {
		return "", fmt.Errorf("failed to create session: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	idleChan := c.startEventListener(ctx, session.ID)

	time.Sleep(100 * time.Millisecond)

	if err := c.sendMessageAsync(session.ID, prompt); err != nil {
		return "", fmt.Errorf("failed to send message: %w", err)
	}

	select {
	case <-idleChan:
	case <-ctx.Done():
	}

	messages, err := c.getMessages(session.ID)
	if err != nil {
		return "", fmt.Errorf("failed to get messages: %w", err)
	}

	response := c.extractAssistantResponse(messages)
	if response == "" {
		if err := c.waitForIdleWithPolling(session.ID, 30*time.Second); err != nil {
			return "", fmt.Errorf("no response received from AI: %w", err)
		}

		messages, err = c.getMessages(session.ID)
		if err != nil {
			return "", fmt.Errorf("failed to get messages: %w", err)
		}

		response = c.extractAssistantResponse(messages)
		if response == "" {
			return "", fmt.Errorf("no response received from AI")
		}
	}

	return response, nil
}

func (c *Client) TestConnection() error {
	req, err := http.NewRequest("GET", c.baseURL+"/global/health", nil)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect to OpenCode server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("OpenCode server returned status %d", resp.StatusCode)
	}

	return nil
}
