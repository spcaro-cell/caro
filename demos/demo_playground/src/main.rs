use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone)]
struct User {
    id: u32,
    name: String,
    email: String,
}

struct AppState {
    users: Mutex<Vec<User>>,
}

fn is_word_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("doc") || ext.eq_ignore_ascii_case("docx"))
        .unwrap_or(false)
}

fn collect_word_files(root: &Path, current: &Path, word_files: &mut Vec<String>) -> std::io::Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_word_files(root, &path, word_files)?;
        } else if is_word_file(&path) {
            let display_path = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string();
            word_files.push(display_path);
        }
    }
    Ok(())
}

fn list_word_files_in_dir<P: AsRef<Path>>(root_dir: P) -> std::io::Result<Vec<String>> {
    let root = root_dir.as_ref();
    let mut word_files = Vec::new();
    collect_word_files(root, root, &mut word_files)?;
    word_files.sort();
    Ok(word_files)
}

#[get("/")]
async fn index() -> impl Responder {
    HttpResponse::Ok().body("Welcome to the API!")
}

#[get("/word-files")]
async fn get_word_files() -> impl Responder {
    match list_word_files_in_dir(".") {
        Ok(files) => HttpResponse::Ok().json(files),
        Err(err) => HttpResponse::InternalServerError()
            .body(format!("Failed to list Word files: {}", err)),
    }
}

#[get("/users")]
async fn get_users(data: web::Data<AppState>) -> impl Responder {
    let users = data.users.lock().unwrap();
    HttpResponse::Ok().json(users.clone())
}

#[post("/users")]
async fn create_user(
    user: web::Json<User>,
    data: web::Data<AppState>,
) -> impl Responder {
    let mut users = data.users.lock().unwrap();
    users.push(user.into_inner());
    HttpResponse::Created().body("User created")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let app_state = web::Data::new(AppState {
        users: Mutex::new(vec![]),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(index)
            .service(get_word_files)
            .service(get_users)
            .service(create_user)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn write_test_file(root: &Path, name: &str) {
        let path = root.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, b"test").unwrap();
    }

    #[test]
    fn list_word_files_in_dir_returns_doc_and_docx() {
        let temp_dir = std::env::temp_dir().join(format!("caro_word_files_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        write_test_file(&temp_dir, "one.docx");
        write_test_file(&temp_dir, "two.doc");
        write_test_file(&temp_dir, "ignore.txt");
        write_test_file(&temp_dir.join("nested"), "nested.docx");

        let mut files = list_word_files_in_dir(&temp_dir).unwrap();
        files.sort();

        assert_eq!(files, vec![
            "nested/nested.docx".to_string(),
            "one.docx".to_string(),
            "two.doc".to_string(),
        ]);

        fs::remove_dir_all(&temp_dir).unwrap();
    }
}
