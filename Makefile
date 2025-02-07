
module.tar.gz:
	go build -a -o module main.go
	tar -czf $@ module

clean:
	rm -rf module module.tar.gz

test:
	go test -v ./...

lint:
	golangci-lint run --timeout 10m