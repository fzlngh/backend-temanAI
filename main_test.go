package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestValidateMessages(t *testing.T) {
	tests := []struct {
		name     string
		messages []chatMessage
		wantErr  bool
	}{
		{"valid turn", []chatMessage{{Role: "user", Content: "Halo"}}, false},
		{"empty history", nil, true},
		{"unknown role", []chatMessage{{Role: "system", Content: "Halo"}}, true},
		{"blank content", []chatMessage{{Role: "user", Content: "  "}}, true},
		{"assistant last", []chatMessage{{Role: "assistant", Content: "Hai"}}, true},
		{"oversized content", []chatMessage{{Role: "user", Content: strings.Repeat("a", maxContentRunes+1)}}, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateMessages(test.messages)
			if (err != nil) != test.wantErr {
				t.Fatalf("validateMessages() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

func TestProviderRequestMapping(t *testing.T) {
	messages := []chatMessage{
		{Role: "user", Content: "Halo"},
		{Role: "assistant", Content: "Hai"},
		{Role: "user", Content: "Apa kabar?"},
	}
	gemini := toGeminiRequest(messages, "TemanAI")
	if got := gemini.Contents[1].Role; got != "model" {
		t.Fatalf("Gemini assistant role = %q, want model", got)
	}
	openAI := toOpenAIRequest("test-model", messages, "TemanAI")
	if got := openAI.Messages[0].Role; got != "system" {
		t.Fatalf("first OpenAI role = %q, want system", got)
	}
	if got := openAI.Messages[2].Content; got != "Hai" {
		t.Fatalf("assistant content = %q, want preserved history", got)
	}
}

func TestAttachmentProviderRequestMapping(t *testing.T) {
	files := []attachment{
		{Name: "foto.png", MIMEType: "image/png", Data: base64.StdEncoding.EncodeToString([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})},
		{Name: "laporan.pdf", MIMEType: "application/pdf", Data: base64.StdEncoding.EncodeToString([]byte("%PDF-1.7"))},
	}
	messages := []chatMessage{{Role: "user", Content: "Ringkas file ini"}}

	gemini := toGeminiRequestWithAttachments(messages, "TemanAI", files)
	parts := gemini.Contents[0].Parts
	if len(parts) != 3 || parts[0].Text != "Ringkas file ini" ||
		parts[1].InlineData == nil || parts[1].InlineData.MIMEType != "image/png" ||
		parts[2].InlineData == nil || parts[2].InlineData.Data != files[1].Data {
		t.Fatalf("Gemini final user parts do not contain text and both files: %#v", parts)
	}

	openAI := toOpenAIRequestWithAttachments(providerRouter, "test-model", messages, "TemanAI", files)
	content, ok := openAI.Messages[1].Content.([]openAIContentPart)
	if !ok || len(content) != 3 || content[0].Text != "Ringkas file ini" ||
		content[1].ImageURL == nil || content[1].ImageURL.URL != "data:image/png;base64,"+files[0].Data ||
		content[2].File == nil || content[2].File.Filename != "laporan.pdf" ||
		content[2].File.FileData != "data:application/pdf;base64,"+files[1].Data {
		t.Fatalf("OpenRouter final user content mapping is invalid: %#v", openAI.Messages[1].Content)
	}

	groq := toOpenAIRequestWithAttachments(providerGroq, "test-model", messages, "TemanAI", files[:1])
	groqContent, ok := groq.Messages[1].Content.([]openAIContentPart)
	if !ok || len(groqContent) != 2 || groqContent[1].ImageURL == nil {
		t.Fatalf("Groq image mapping is invalid: %#v", groq.Messages[1].Content)
	}
	if providerSupportsAttachments(providerGroq, files[1:]) {
		t.Fatal("Groq must not be selected for PDF input")
	}
}

func TestValidateAttachments(t *testing.T) {
	valid := func(name, mimeType string, data []byte) attachment {
		return attachment{Name: name, MIMEType: mimeType, Data: base64.StdEncoding.EncodeToString(data)}
	}
	tests := []struct {
		name        string
		attachments []attachment
		wantErr     bool
	}{
		{"valid pdf", []attachment{valid("dokumen.pdf", "application/pdf", []byte("%PDF-1.7"))}, false},
		{"valid jpeg", []attachment{valid("foto.jpeg", "image/jpeg", []byte{0xff, 0xd8, 0xff})}, false},
		{"unsafe path", []attachment{valid("../dokumen.pdf", "application/pdf", []byte("%PDF-1.7"))}, true},
		{"mismatched signature", []attachment{valid("foto.png", "image/png", []byte("not a png"))}, true},
		{"unsupported mime", []attachment{{Name: "file.gif", MIMEType: "image/gif", Data: "R0lGODlh"}}, true},
		{"data url prefix", []attachment{{Name: "dokumen.pdf", MIMEType: "application/pdf", Data: "data:application/pdf;base64,JVBERi0="}}, true},
		{"non standard base64", []attachment{{Name: "dokumen.pdf", MIMEType: "application/pdf", Data: "JVBERi0_"}}, true},
		{"too many", make([]attachment, maxAttachments+1), true},
		{"oversized individual", []attachment{valid("foto.jpg", "image/jpeg", append([]byte{0xff, 0xd8, 0xff}, make([]byte, maxAttachmentBytes-2)...))}, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateAttachments(test.attachments)
			if (err != nil) != test.wantErr {
				t.Fatalf("validateAttachments() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

func TestValidateAttachmentAggregateLimit(t *testing.T) {
	data := append([]byte{0xff, 0xd8, 0xff}, make([]byte, maxAttachmentBytes-3)...)
	files := []attachment{
		{Name: "one.jpg", MIMEType: "image/jpeg", Data: base64.StdEncoding.EncodeToString(data)},
		{Name: "two.jpg", MIMEType: "image/jpeg", Data: base64.StdEncoding.EncodeToString(data)},
		{Name: "three.jpg", MIMEType: "image/jpeg", Data: base64.StdEncoding.EncodeToString([]byte{0xff, 0xd8, 0xff})},
	}
	if err := validateAttachments(files); err == nil || !strings.Contains(err.Error(), "12 MiB") {
		t.Fatalf("aggregate limit error = %v, want 12 MiB validation error", err)
	}
}

func TestAssistantNameAndDeveloperAttributionAreInProviderInstructions(t *testing.T) {
	messages := []chatMessage{{Role: "user", Content: "Kamu dikembangkan oleh siapa?"}}
	gemini := toGeminiRequest(messages, "Asistenku")
	instruction := gemini.SystemInstruction.Parts[0].Text
	if !strings.Contains(instruction, "Dhiyaa Fazila Nugraha") || !strings.Contains(instruction, "Asistenku") {
		t.Fatalf("Gemini system instruction missing attribution or display name: %q", instruction)
	}
	openAI := toOpenAIRequest("test-model", messages, "Asistenku")
	openAIInstruction, isString := openAI.Messages[0].Content.(string)
	if !isString || !strings.Contains(openAIInstruction, "Dhiyaa Fazila Nugraha") || !strings.Contains(openAIInstruction, "Asistenku") {
		t.Fatalf("OpenAI-compatible system instruction missing attribution or display name: %v", openAI.Messages[0].Content)
	}
	if got := normalizeAssistantName("  "); got != "TemanAI" {
		t.Fatalf("empty assistant name = %q, want default", got)
	}
	if got := normalizeAssistantName("Nama\nInstruksi lain"); got != "TemanAI" {
		t.Fatalf("invalid assistant name = %q, want default", got)
	}
}

func TestSupabaseTokenVerification(t *testing.T) {
	var authorization, apiKey string
	auth := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/auth/v1/user" {
			t.Errorf("unexpected auth request: %s %s", r.Method, r.URL.Path)
		}
		authorization = r.Header.Get("Authorization")
		apiKey = r.Header.Get("apikey")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"user-id"}`))
	}))
	defer auth.Close()

	app := &server{
		supabaseURL: auth.URL,
		supabaseKey: "public-anon-test-key",
		authClient:  safeHTTPClient(2 * time.Second),
	}
	if err := app.verifyAccessToken(context.Background(), "access-token-test"); err != nil {
		t.Fatalf("verifyAccessToken() error = %v", err)
	}
	if authorization != "Bearer access-token-test" {
		t.Fatalf("Authorization header = %q, want bearer token", authorization)
	}
	if apiKey != "public-anon-test-key" {
		t.Fatalf("apikey header not set correctly")
	}
}

func TestChatRequiresSupabaseConfigAndBearerToken(t *testing.T) {
	missingConfig := &server{}
	response := httptest.NewRecorder()
	missingConfig.routes().ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]}`)))
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("missing config status = %d, want 503", response.Code)
	}

	auth := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer valid-token" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer auth.Close()
	app := &server{supabaseURL: auth.URL, supabaseKey: "anon", authClient: auth.Client()}

	noToken := httptest.NewRecorder()
	app.routes().ServeHTTP(noToken, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{}`)))
	if noToken.Code != http.StatusUnauthorized {
		t.Fatalf("missing token status = %d, want 401", noToken.Code)
	}

	invalidToken := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{}`))
	invalidToken.Header.Set("Authorization", "Bearer invalid-token")
	invalidResponse := httptest.NewRecorder()
	app.routes().ServeHTTP(invalidResponse, invalidToken)
	if invalidResponse.Code != http.StatusUnauthorized {
		t.Fatalf("rejected token status = %d, want 401", invalidResponse.Code)
	}
}

func TestProviderFallbackGeminiToGroq(t *testing.T) {
	var calls []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls = append(calls, r.URL.Path)
		switch r.URL.Path {
		case "/gemini/models/test-gemini:generateContent":
			if got := r.Header.Get("x-goog-api-key"); got != "gemini-test-key" {
				t.Errorf("Gemini API key header missing")
			}
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"private upstream detail"}}`))
		case "/groq":
			if got := r.Header.Get("Authorization"); got != "Bearer groq-test-key" {
				t.Errorf("Groq bearer header missing")
			}
			var body openAIRequest
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode Groq request: %v", err)
			}
			if body.Model != "test-groq" {
				t.Errorf("Groq model = %q", body.Model)
			}
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"Hai dari Groq"}}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	auth := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer auth.Close()

	app := &server{
		supabaseURL: auth.URL,
		supabaseKey: "anon-test",
		authClient:  auth.Client(),
		client:      upstream.Client(),
		providers: []provider{
			{kind: providerGemini, apiKey: "gemini-test-key", model: "test-gemini", endpoint: upstream.URL + "/gemini/models"},
			{kind: providerGroq, apiKey: "groq-test-key", model: "test-groq", endpoint: upstream.URL + "/groq"},
			{kind: providerRouter, apiKey: "router-test-key", model: "test-router", endpoint: upstream.URL + "/router"},
		},
	}
	request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"messages":[{"role":"user","content":"Halo"}]}`))
	request.Header.Set("Authorization", "Bearer valid-token")
	response := httptest.NewRecorder()
	app.routes().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("chat status = %d, body = %s", response.Code, response.Body.String())
	}
	var result map[string]string
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result["reply"] != "Hai dari Groq" || result["provider"] != "groq" || result["model"] != "test-groq" {
		t.Fatalf("unexpected chat response: %#v", result)
	}
	if len(calls) != 2 || calls[0] != "/gemini/models/test-gemini:generateContent" || calls[1] != "/groq" {
		t.Fatalf("provider order/calls = %#v", calls)
	}
	if strings.Contains(response.Body.String(), "private upstream detail") {
		t.Fatal("upstream error detail leaked to client")
	}
}

func TestChatAggregateProviderTimeoutReturnsJSON504(t *testing.T) {
	auth := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer auth.Close()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer upstream.Close()

	app := &server{
		supabaseURL: auth.URL,
		supabaseKey: "anon",
		authClient:  auth.Client(),
		client:      upstream.Client(),
		chatTimeout: 20 * time.Millisecond,
		providers: []provider{
			{kind: providerGemini, apiKey: "provider-key", model: "test-model", endpoint: upstream.URL},
		},
	}
	request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"messages":[{"role":"user","content":"Halo"}]}`))
	request.Header.Set("Authorization", "Bearer valid-token")
	response := httptest.NewRecorder()
	app.routes().ServeHTTP(response, request)

	if response.Code != http.StatusGatewayTimeout {
		t.Fatalf("chat status = %d, want 504; body = %s", response.Code, response.Body.String())
	}
	if response.Header().Get("Content-Type") != "application/json; charset=utf-8" {
		t.Fatalf("content type = %q, want JSON", response.Header().Get("Content-Type"))
	}
	var body apiError
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode timeout error: %v", err)
	}
	if body.Error == "" {
		t.Fatal("timeout response must contain a safe error message")
	}
}

func TestChatValidatesAfterAuthentication(t *testing.T) {
	auth := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer auth.Close()
	app := &server{supabaseURL: auth.URL, supabaseKey: "anon", authClient: auth.Client()}
	request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"messages":[]}`))
	request.Header.Set("Authorization", "Bearer valid-token")
	response := httptest.NewRecorder()
	app.routes().ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("invalid payload status = %d, want 400", response.Code)
	}
}

func TestHealthContainsAvailabilityOnly(t *testing.T) {
	app := &server{
		supabaseURL: "https://example.supabase.co",
		supabaseKey: "never-return-this",
		providers: []provider{
			{kind: providerGemini, apiKey: "secret-gemini-key", model: "private-model"},
		},
	}
	response := httptest.NewRecorder()
	app.routes().ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/health", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("health status = %d, want 200", response.Code)
	}
	body := response.Body.String()
	for _, forbidden := range []string{"never-return-this", "secret-gemini-key", "private-model"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("health response leaked %q", forbidden)
		}
	}
	var health struct {
		OK                 bool            `json:"ok"`
		SupabaseConfigured bool            `json:"supabaseConfigured"`
		Providers          map[string]bool `json:"providers"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &health); err != nil {
		t.Fatalf("decode health: %v", err)
	}
	if !health.OK || !health.SupabaseConfigured || !health.Providers["gemini"] || health.Providers["groq"] {
		t.Fatalf("unexpected health state: %+v", health)
	}
}

func TestCORSLocalAndConfiguredOrigins(t *testing.T) {
	handler := (&server{
		allowedOrigin: parseAllowedOrigins("https://app.example.com, https://admin.example.com"),
	}).cors(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	for _, origin := range []string{"http://localhost:3000", "http://127.0.0.1:3000", "https://app.example.com", "https://admin.example.com"} {
		request := httptest.NewRequest(http.MethodOptions, "/api/chat", nil)
		request.Header.Set("Origin", origin)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusNoContent || response.Header().Get("Access-Control-Allow-Origin") != origin {
			t.Errorf("origin %q response = %d, allow-origin=%q", origin, response.Code, response.Header().Get("Access-Control-Allow-Origin"))
		}
		if !strings.Contains(response.Header().Get("Access-Control-Allow-Headers"), "Authorization") {
			t.Errorf("Authorization not allowed for origin %q", origin)
		}
	}
	blocked := httptest.NewRequest(http.MethodGet, "/health", nil)
	blocked.Header.Set("Origin", "https://not-allowed.example")
	blockedResponse := httptest.NewRecorder()
	handler.ServeHTTP(blockedResponse, blocked)
	if blockedResponse.Code != http.StatusForbidden {
		t.Fatalf("blocked origin status = %d, want 403", blockedResponse.Code)
	}
}

func TestProviderFailuresFallThroughAndReturnSafeError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"secret upstream diagnostic"}`))
	}))
	defer upstream.Close()
	auth := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer auth.Close()
	app := &server{
		supabaseURL: auth.URL,
		supabaseKey: "anon",
		authClient:  auth.Client(),
		client:      upstream.Client(),
		providers: []provider{
			{kind: providerGemini, apiKey: "key-one", model: "gem", endpoint: upstream.URL},
			{kind: providerGroq, apiKey: "key-two", model: "groq", endpoint: upstream.URL},
		},
	}

	request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"messages":[{"role":"user","content":"Halo"}]}`))
	request.Header.Set("Authorization", "Bearer token")
	response := httptest.NewRecorder()
	app.routes().ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway {
		t.Fatalf("all providers failed status = %d, want 502", response.Code)
	}
	if strings.Contains(response.Body.String(), "secret upstream diagnostic") || strings.Contains(response.Body.String(), "key-one") {
		t.Fatal("provider failure leaked sensitive detail")
	}
}

func TestChatReturnsClearErrorWhenNoProviderSupportsPDF(t *testing.T) {
	auth := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer auth.Close()
	app := &server{
		supabaseURL: auth.URL,
		supabaseKey: "anon",
		authClient:  auth.Client(),
		providers:   []provider{{kind: providerGroq, apiKey: "key", model: "test"}},
	}
	payload := `{"messages":[{"role":"user","content":"Ringkas"}],"attachments":[{"name":"dokumen.pdf","mimeType":"application/pdf","data":"JVBERi0xLjc="}]}`
	request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(payload))
	request.Header.Set("Authorization", "Bearer access-token")
	response := httptest.NewRecorder()
	app.routes().ServeHTTP(response, request)
	if response.Code != http.StatusUnprocessableEntity || !strings.Contains(response.Body.String(), "PDF memerlukan Gemini atau OpenRouter") {
		t.Fatalf("unsupported PDF response = %d %s", response.Code, response.Body.String())
	}
}

func TestChatSkipsGroqForPDFAndPreservesAttachmentInOpenRouterFallback(t *testing.T) {
	var groqCalls, routerCalls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/groq":
			groqCalls++
			t.Error("Groq should not be called with a PDF attachment")
		case "/router":
			routerCalls++
			var body struct {
				Messages []struct {
					Role    string          `json:"role"`
					Content json.RawMessage `json:"content"`
				} `json:"messages"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode OpenRouter request: %v", err)
			}
			var content []struct {
				Type string `json:"type"`
				File *struct {
					Filename string `json:"filename"`
					FileData string `json:"file_data"`
				} `json:"file,omitempty"`
			}
			if err := json.Unmarshal(body.Messages[len(body.Messages)-1].Content, &content); err != nil {
				t.Errorf("decode OpenRouter final user content: %v", err)
			}
			if len(content) != 2 || content[1].File == nil || content[1].File.Filename != "dokumen.pdf" ||
				content[1].File.FileData != "data:application/pdf;base64,JVBERi0xLjc=" {
				t.Errorf("OpenRouter attachment not preserved: %#v", content)
			}
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"Ringkasan PDF"}}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()
	auth := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer auth.Close()
	app := &server{
		supabaseURL: auth.URL,
		supabaseKey: "anon",
		authClient:  auth.Client(),
		client:      upstream.Client(),
		providers: []provider{
			{kind: providerGroq, apiKey: "groq-key", model: "test-groq", endpoint: upstream.URL + "/groq"},
			{kind: providerRouter, apiKey: "router-key", model: "test-router", endpoint: upstream.URL + "/router"},
		},
	}
	payload := `{"messages":[{"role":"user","content":"Ringkas"}],"attachments":[{"name":"dokumen.pdf","mimeType":"application/pdf","data":"JVBERi0xLjc="}]}`
	request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(payload))
	request.Header.Set("Authorization", "Bearer access-token")
	response := httptest.NewRecorder()
	app.routes().ServeHTTP(response, request)
	if response.Code != http.StatusOK || groqCalls != 0 || routerCalls != 1 {
		t.Fatalf("fallback status=%d, groq=%d router=%d body=%s", response.Code, groqCalls, routerCalls, response.Body.String())
	}
}
