// Apple on-device Translation framework bridge for the local VERGE dashboard.
// Compile with `swiftc -o <temporary executable> apple_translate.swift` on macOS 26+.
// Input text is read only from stdin, never passed as a process argument or saved to disk.
import Foundation
import NaturalLanguage
import Translation

@main
struct AppleTranslate {
    static func main() async {
        guard let text = String(data: FileHandle.standardInput.readDataToEndOfFile(), encoding: .utf8),
              !text.isEmpty else {
            fputs("EMPTY_INPUT\n", stderr)
            exit(2)
        }
        let sourceCode = NLLanguageRecognizer.dominantLanguage(for: text)?.rawValue ?? "en"
        let source = Locale.Language(identifier: sourceCode)
        let target = Locale.Language(identifier: "zh-Hans")
        if source.languageCode == target.languageCode {
            print(text)
            return
        }
        do {
            let session = TranslationSession(installedSource: source, target: target)
            let response = try await session.translate(text)
            print(response.targetText)
        } catch TranslationError.notInstalled {
            fputs("LANGUAGE_PACK_NOT_INSTALLED\n", stderr)
            exit(3)
        } catch {
            // Do not print the error description; some framework errors may include input text.
            fputs("TRANSLATION_FAILED\n", stderr)
            exit(4)
        }
    }
}
